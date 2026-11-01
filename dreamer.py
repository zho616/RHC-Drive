import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)

        context_config = getattr(config, "context", None)
        history_flag = getattr(context_config, "use_history_context", None) if context_config else None
        self.context_enabled = bool(context_config and context_config.enabled)
        self.context_source = str(getattr(context_config, "source", "encoded_history")) if context_config else "encoded_history"
        if self.context_source not in ("encoded_history", "oracle", "raw_history"):
            raise ValueError("Unknown context source")
        self.context_dim = 8
        if history_flag is not None and bool(history_flag) != self.context_enabled:
            raise ValueError("context.enabled and use_history_context must agree (link enabled to the switch in ablation configs)")
        self.context_response_enabled = self.context_enabled and bool(getattr(context_config, "use_response_loss", True))
        self.context_prediction_horizon = getattr(context_config, "prediction_horizon", 5) if context_config else 5
        if type(self.context_prediction_horizon) is not int or self.context_prediction_horizon < 1:
            raise ValueError("context.prediction_horizon must be a positive integer")
        self.context_log_grads = self.context_enabled and bool(getattr(context_config, "log_grad_norms", False))
        self.context_shallow = self.context_enabled and context_config.injection == "dali_shallow"
        self.context_wm_grad_scale = float(getattr(context_config, "wm_grad_scale", 0.0)) if context_config else 0.0
        if not math.isfinite(self.context_wm_grad_scale) or self.context_wm_grad_scale < 0:
            raise ValueError("context.wm_grad_scale must be finite and nonnegative")
        if self.context_wm_grad_scale and not (self.context_shallow and self.rep_loss == "r2dreamer"):
            raise ValueError("context.wm_grad_scale requires R2Dreamer dali_shallow")
        self.context_deep = self.context_enabled and context_config.injection in ("dali_deep", "dali_deep_joint")
        self.context_actor_critic = self.context_enabled and not self.context_shallow and (
            self.context_deep or bool(getattr(context_config, "actor_critic", False)))
        rssm_class = rssm.RSSM
        rssm_kwargs = {}
        if self.context_enabled:
            from context_rssm import ContextRSSM
            if self.rep_loss not in ("r2dreamer", "dreamer") or self.act_dim != 2 or hasattr(act_space, "n") or hasattr(act_space, "multi_discrete"):
                raise ValueError("Context supports continuous two-action R2Dreamer/DreamerV3 only")
            if context_config.injection not in ("recurrent_concat", "prior_posterior", "adarms_deter", "dali_deep", "dali_deep_joint", "dali_shallow"):
                raise ValueError("Unknown context injection")
            rssm_kwargs["injection"] = str(context_config.injection)
            rssm_kwargs["context_input_grad"] = self.context_wm_grad_scale > 0
            self.context_rollout = str(context_config.rollout)
            if self.context_rollout not in ("dynamic", "fixed", "zero", "shuffled"):
                raise ValueError("Unknown context rollout mode")
            required = {"ctx_ego_state", "ctx_prev_action", "ctx_history", "ctx_mask", "ctx_collision"}
            if not required.issubset(obs_space.spaces):
                raise ValueError("Context requires MetaDrive physical history observations")
            self.context_state_dim = int(obs_space.spaces["ctx_ego_state"].shape[-1])
            if self.context_state_dim not in (2, 4):
                raise ValueError("Context expects legacy state2 or speed/wheel-angle state4")
            expected = {"ctx_ego_state": (self.context_state_dim,), "ctx_prev_action": (2,),
                        "ctx_history": (5, self.context_state_dim + 2), "ctx_mask": (5,)}
            if any(tuple(obs_space.spaces[k].shape) != shape for k, shape in expected.items()):
                raise ValueError("Context state/history shapes disagree; use matching replay and checkpoint layouts")
            if self.context_source != "encoded_history":
                if not self.context_shallow or self.context_wm_grad_scale or self.context_response_enabled:
                    raise ValueError("Direct conditions require Shallow, no auxiliary losses, and no encoder gradients")
                from context import ORACLE_KEYS
                if self.context_source == "oracle" and not set(ORACLE_KEYS).issubset(obs_space.spaces):
                    raise ValueError("Oracle conditioning requires actual chassis parameters")
                self.context_dim = len(ORACLE_KEYS) if self.context_source == "oracle" else 5 * (self.context_state_dim + 2) + 5
                rssm_kwargs["context_dim"] = self.context_dim
            rssm_class = ContextRSSM

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm_class(
            config.rssm,
            self.embed_size,
            self.act_dim,
            **rssm_kwargs,
        )
        prediction_size = self.rssm.feat_size + (8 if self.context_deep else 0)
        self.reward = networks.MLPHead(config.reward, prediction_size)
        self.cont = networks.MLPHead(config.cont, prediction_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        behavior_size = self.rssm.feat_size + (8 if self.context_actor_critic else 0)
        self.actor = networks.MLPHead(config.actor, behavior_size)
        self.value = networks.MLPHead(config.critic, behavior_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter + (8 if self.context_deep else 0),
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(prediction_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        if self.context_enabled:
            from context import HistoryEncoder, ResidualDynamics
            if self.context_source == "encoded_history":
                self.context_encoder = HistoryEncoder(self.context_state_dim)
            aux_weight = float(context_config.aux_weight)
            if not math.isfinite(aux_weight) or aux_weight < 0:
                raise ValueError("context.aux_weight must be finite and nonnegative")
            if self.context_response_enabled:
                self._loss_scales["context_aux"] = aux_weight
            self.context_swap_weight = float(getattr(context_config, "swap_weight", 0.0))
            if not math.isfinite(self.context_swap_weight) or self.context_swap_weight < 0:
                raise ValueError("context.swap_weight must be finite and nonnegative")
            exchange_flag = getattr(context_config, "use_context_exchange", None)
            self.context_exchange_enabled = bool(self.context_swap_weight) if exchange_flag is None else bool(exchange_flag)
            if self.context_source != "encoded_history" and (self.context_exchange_enabled or aux_weight or self.context_swap_weight):
                raise ValueError("Direct conditions must disable response and exchange weights")
            if self.context_exchange_enabled and self.context_swap_weight <= 0:
                raise ValueError("Enabled Context Exchange requires a positive swap_weight")
            if self.context_exchange_enabled:
                from context import SWAP_PROFILE_KEYS
                if not set((*SWAP_PROFILE_KEYS, "ctx_episode_uid")).issubset(obs_space.spaces):
                    raise ValueError("Context exchange requires chassis metadata and true episode IDs")
                self._loss_scales["context_swap"] = self.context_swap_weight
            if hasattr(self, "context_encoder"):
                modules.update(context_encoder=self.context_encoder)
            if self.context_response_enabled or self.context_exchange_enabled:
                self.context_dynamics = ResidualDynamics(self.context_state_dim)
                modules.update(context_dynamics=self.context_dynamics)
            elif self.context_source == "encoded_history" and (not self.context_shallow or not self.context_wm_grad_scale):
                raise ValueError("History-only requires dali_shallow and positive wm_grad_scale; otherwise E has no training signal")
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler(enabled=self.device.type == 'cuda')

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        # (B, S, K), (B, D)
        extra = ()
        if self.context_enabled:
            context = self._context_condition(p_obs)
            context = torch.where(obs["is_first"].bool().reshape(-1, 1), 0., context)
            extra = (context,)
        stoch, deter, _ = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"], *extra)
        # (B, F)
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(self._behavior_feat(feat, extra[0] if extra else None))
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        if self.context_enabled:
            raise ValueError("Context video prediction is unsupported; use vector MetaDrive evaluation")
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.context_log_grads:
            # Actual unscaled gradients before AGC; diagnostic only, no graph or RNG changes.
            for name in ("context_encoder", "context_dynamics", "rssm"):
                module = getattr(self, name, None)
                grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None] if module is not None else []
                mets[f"grad/{name}_norm"] = torch.stack(grads).norm() if grads else torch.zeros((), device=self.device)
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        extra = ()
        if self.context_enabled:
            context = self._context_condition(data)
            context = torch.where(data["is_first"].bool().reshape(B, T, 1), 0., context)
            if self.context_response_enabled:
                from context import auxiliary_loss
                losses["context_aux"], metrics["context/valid_targets"], physical_metrics = auxiliary_loss(
                    self.context_dynamics, data, context, return_metrics=True,
                    prediction_horizon=self.context_prediction_horizon)
                metrics.update(physical_metrics)
            if self.context_exchange_enabled:
                from context import context_swap_loss
                losses["context_swap"], swap_metrics = context_swap_loss(self.context_dynamics, data, context)
                metrics.update(swap_metrics)
            metrics["context/rms"] = context.detach().square().mean().sqrt()
            # A separate leaf prevents auxiliary losses from reaching WM parameters
            # and repval from reaching E. Only the selected WM gradient is routed below.
            wm_context = context.detach().requires_grad_(self.context_wm_grad_scale > 0)
            extra = (wm_context,)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"], *extra)
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter, *extra)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        prediction_feat = self._prediction_feat(feat, context if self.context_enabled else None)
        if self.rep_loss == "dreamer":
            decoder_deter = self._prediction_feat(post_deter, context if self.context_enabled else None)
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, decoder_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(prediction_feat.reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # reward and continue
        losses["rew"] = torch.mean(-self.reward(prediction_feat).log_prob(to_f32(data["reward"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(prediction_feat).log_prob(cont))
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        if self.context_enabled:
            imag_feat, imag_action, imag_context = self._imagine_context(start, self.imag_horizon + 1, data, context.detach())
        else:
            imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
        imag_behavior_feat = self._behavior_feat(imag_feat, imag_context if self.context_enabled else None)

        # (B*T, T_imag, 1)
        imag_prediction_feat = self._prediction_feat(imag_feat, imag_context if self.context_enabled else None)
        imag_reward = self._frozen_reward(imag_prediction_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_prediction_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_behavior_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_behavior_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_behavior_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_behavior_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        behavior_feat = self._behavior_feat(feat, context if self.context_enabled else None)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(behavior_feat).mode()
        slow_value = self._frozen_slow_value(behavior_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(behavior_feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        if self.context_wm_grad_scale:
            surrogate, context_metrics = self._context_wm_surrogate(losses, context, wm_context)
            total_loss = total_loss + surrogate
            metrics.update(context_metrics)
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    CONTEXT_WM_LOSSES = ("dyn", "rep", "barlow", "rew", "con")

    def _context_condition(self, data):
        if self.context_source == "encoded_history":
            return self.context_encoder(data["ctx_history"], data["ctx_mask"])
        from context import direct_condition
        return direct_condition(data, self.context_source, self.context_state_dim)

    def _context_wm_surrogate(self, losses, context, wm_context):
        """Route selected first-order WM gradients to E, preserving all WM gradients.

        The zero-valued surrogate has no connection to WM or auxiliary predictor
        parameters. Scaling before the VJP preserves GradScaler's AMP protection.
        repval/policy/value are intentionally absent from this whitelist.
        """
        objective = sum(losses[k] * self._loss_scales[k] for k in self.CONTEXT_WM_LOSSES)
        gradient, = torch.autograd.grad(self._scaler.scale(objective), wm_context, retain_graph=True)
        gradient = gradient.detach() / self._scaler.get_scale()
        surrogate = self.context_wm_grad_scale * ((context - context.detach()) * gradient).sum()
        return surrogate, {"context/wm_input_grad_norm": gradient.float().norm(),
                           "context/wm_grad_scale": context.new_tensor(self.context_wm_grad_scale)}

    def _prediction_feat(self, feat, context=None):
        """DALI Deep conditions prediction heads; context learns only from auxiliaries."""
        if not self.context_deep:
            return feat
        if context is None or context.shape[:-1] != feat.shape[:-1]:
            raise ValueError("Prediction heads require time-aligned context")
        return torch.cat((feat, context.detach().to(feat.dtype)), -1)

    def _behavior_feat(self, feat, context=None):
        """Condition decisions on aligned context without training E through RL."""
        if not self.context_actor_critic:
            return feat
        if context is None or context.shape[:-1] != feat.shape[:-1]:
            raise ValueError("Actor/critic requires time-aligned context")
        return torch.cat((feat, context.detach().to(feat.dtype)), -1)

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _imagine_context(self, start, horizon, data, context):
        from context import refresh_context
        state = data["ctx_ego_state"].reshape(-1, self.context_state_dim)
        history = data["ctx_history"].reshape(-1, 5, self.context_state_dim + 2)
        mask = data["ctx_mask"].reshape(-1, 5).bool()
        context = context.reshape(-1, self.context_dim)
        fixed = context
        alive = ~(data["is_last"] | data["is_terminal"]).bool().reshape(-1)
        stoch, deter = start
        feats, actions, contexts = [], [], []
        behavior_context = context
        for _ in range(horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(self._behavior_feat(feat, behavior_context)).rsample()
            feats.append(feat)
            actions.append(action)
            contexts.append(behavior_context)
            if hasattr(self, "context_dynamics"):
                state, context, history, mask = refresh_context(
                    self.context_encoder, self.context_dynamics, state, action, context, history, mask, alive)
            # History-only has no response predictor. In Shallow, future context
            # is unused by transition/prior, prediction heads and actor/critic.
            injected = context
            if self.context_rollout == "fixed":
                injected = torch.where(alive[:, None], fixed, 0.)
            elif self.context_rollout == "zero":
                injected = torch.zeros_like(context)
            elif self.context_rollout == "shuffled":
                injected = context[torch.randperm(context.shape[0], device=context.device)]
                injected = torch.where(alive[:, None], injected, 0.)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action, injected)
            behavior_context = injected
            next_feat = self._frozen_rssm.get_feat(stoch, deter)
            alive = alive & (self._frozen_cont(self._prediction_feat(next_feat, injected)).mean.reshape(-1) > 0.5)
        return torch.stack(feats, 1), torch.stack(actions, 1), torch.stack(contexts, 1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
