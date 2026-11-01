"""Frozen Shallow-context interventions, inspired by DALI Section 6.3.

Perturb each context coordinate by one corpus standard deviation. Replay the
same causal observation window through the ORIGINAL posterior, then imagine
50 zero-action steps. No decoder, physical probe, or world-model weight is fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--samples', type=int, default=2500)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=125)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from dreamer import Dreamer
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cfg = OmegaConf.load(a.model/'.hydra/config.yaml')
    cfg.device = a.device
    OmegaConf.resolve(cfg)
    cfg.model.compile = False
    assert cfg.model.context.injection == 'dali_shallow'
    episodes = json.loads((a.corpus/'episodes.json').read_text())
    data = [dict(np.load(a.corpus/'episodes'/e['file'])) for e in episodes]
    excluded = {'executed_action', 'speed', 'wheel_degrees', 'reward'}
    spaces = {k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape[1:], dtype=np.float32)
              for k,v in data[0].items() if k not in excluded}
    model = Dreamer(cfg.model, gym.spaces.Dict(spaces),
                    gym.spaces.Box(-1, 1, (2,), np.float32)).to(a.device)
    checkpoint = a.model/'best.pt'
    weight_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location=a.device, weights_only=False)
    model.load_state_dict(payload['agent_state_dict'], strict=True)
    step = int(payload['step'])
    del payload
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    decoder = hasattr(model, 'decoder')
    tensor = lambda x: torch.as_tensor(x, device=a.device)
    K, H = 5, a.horizon
    candidates = []
    for ei,(e,d) in enumerate(zip(episodes,data)):
        # All conditioning observations are at or before the forecast origin.
        for t in range(20, len(d['state'])):
            if d['ctx_mask'][t-K+1:t+1].all() and not d['is_last'][t-K+1:t+1].any() and not d['ctx_collision'][t-K+1:t+1].any():
                candidates.append(dict(episode=ei, t=t, map_id=e['map_id'],
                                       model=e['model'], collector=e['policy']))
        np.testing.assert_allclose(d['state'][:,3]*51-1, d['speed']*3.6, atol=1e-4)
    # Identical origins for the two models; equal map weights and no replacement.
    rng = np.random.default_rng(20261025)
    maps = sorted({o['map_id'] for o in candidates})
    origins = []
    for mi,m in enumerate(maps):
        pool = [o for o in candidates if o['map_id'] == m]
        count = a.samples//len(maps) + (mi < a.samples % len(maps))
        assert count <= len(pool)
        origins.extend(pool[int(i)] for i in rng.choice(len(pool), count, replace=False))
    n = len(origins)
    (a.output/'origins.json').write_text(json.dumps(origins, indent=2))
    lookup = {(o['episode'],o['t']-K+1):i for i,o in enumerate(origins)}
    windows = {}
    contexts_for_scale = []
    with torch.inference_mode():
        for start in range(0,len(data),10):
            ids = list(range(start,min(start+10,len(data))))
            length = max(len(data[i]['state']) for i in ids)
            batch = {}
            for key in ['state','is_first','ctx_history','ctx_mask']:
                arr = np.zeros((len(ids),length,*data[0][key].shape[1:]),data[0][key].dtype)
                for b,i in enumerate(ids):
                    arr[b,:len(data[i][key])] = data[i][key]
                batch[key] = tensor(arr)
            emb = model.encoder(TensorDict(batch,batch_size=[len(ids),length]))
            ctx = model.context_encoder(batch['ctx_history'],batch['ctx_mask'])
            prev = torch.zeros(len(ids),length,2,device=a.device)
            for b,i in enumerate(ids):
                prev[b,1:len(data[i]['state'])] = tensor(data[i]['executed_action'][:-1])
                mask = (np.arange(len(data[i]['state'])) >= 20) & data[i]['ctx_mask'].all(-1) & ~data[i]['is_last'] & ~data[i]['ctx_collision']
                contexts_for_scale.append(ctx[b,:len(mask)][tensor(mask)].cpu().numpy())
            torch.manual_seed(20261025+start)
            z,h = model.rssm.initial(len(ids))
            for t in range(length):
                for b,ei in enumerate(ids):
                    oi = lookup.get((ei,t))
                    if oi is not None:
                        item = dict(z=z[b].clone(),h=h[b].clone(),emb=emb[b,t:t+K].clone(),
                                    ctx=ctx[b,t:t+K].clone(),prev=prev[b,t:t+K].clone(),
                                    reset=batch['is_first'][b,t:t+K].clone())
                        windows[oi] = item
                z,h,_ = model.rssm.obs_step(z,h,prev[:,t],emb[:,t],batch['is_first'][:,t],ctx[:,t])
            print(a.label,'causal filtering',start+len(ids),'/',len(data),flush=True)
        assert len(windows) == n
        sigma = np.concatenate(contexts_for_scale).std(0)
        assert sigma.shape == (8,) and np.isfinite(sigma).all()
        C = 21 if decoder else 2  # two native heads, plus native ego/navigation channels
        predicted = np.empty((9,n,H,C),np.float32)
        posterior_kl = np.zeros((8,n),np.float32)
        prior_js = np.zeros((8,n,H),np.float32)
        initial_latent_change = np.zeros((8,n),bool)
        exact_self_check = True
        exact_future_only_check = True
        def kl(p,q):
            return (p*(p.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum((-1,-2))
        for start in range(0,n,a.batch_size):
            stop = min(start+a.batch_size,n)
            w = {k:torch.stack([windows[i][k] for i in range(start,stop)]) for k in windows[0]}
            base_q,base_z,base_h,base_prior = None,None,None,[]
            for variant in range(10):  # final branch is an exact self-intervention control
                c = w['ctx'].clone()
                if 1 <= variant <= 8:
                    c[:,:,variant-1] += float(sigma[variant-1])
                z,h = w['z'].clone(),w['h'].clone()
                torch.manual_seed(421000+start)
                for t in range(K):
                    z,h,q = model.rssm.obs_step(z,h,w['prev'][:,t],w['emb'][:,t],w['reset'][:,t],c[:,t])
                probs = model.rssm.get_dist(q).base_dist.probs
                if variant == 0:
                    base_q,base_z,base_h = probs.clone(),z.clone(),h.clone()
                    # A Shallow future-only intervention is structurally a null:
                    # identical initial state/action/RNG, different future c.
                    zero = torch.zeros(stop-start,2,device=a.device)
                    torch.manual_seed(9911)
                    null0 = model.rssm.img_step(z.clone(),h.clone(),zero,c[:,-1])
                    pert = c[:,-1].clone();pert[:,0] += float(sigma[0])
                    torch.manual_seed(9911)
                    null1 = model.rssm.img_step(z.clone(),h.clone(),zero,pert)
                    exact_future_only_check &= all(torch.equal(x,y) for x,y in zip(null0,null1))
                elif variant <= 8:
                    posterior_kl[variant-1,start:stop] = kl(base_q,probs).cpu().numpy()
                    initial_latent_change[variant-1,start:stop] = ((z!=base_z).any((-1,-2)) | (h!=base_h).any(-1)).cpu().numpy()
                result = []
                # Reset after the negative-control calls, sharing all future noise.
                torch.manual_seed(821000+start)
                action = torch.zeros(stop-start,2,device=a.device)
                for t in range(H):
                    h = model.rssm._transition(z,h,action,c[:,-1])
                    logits = model.rssm._img_net(h)
                    dist = model.rssm.get_dist(logits)
                    pp = dist.base_dist.probs
                    z = dist.rsample()
                    if variant == 0:
                        base_prior.append(pp.clone())
                    elif variant <= 8:
                        mid = .5*(base_prior[t]+pp)
                        js = .5*(kl(base_prior[t],mid)+kl(pp,mid))
                        prior_js[variant-1,start:stop,t] = js.clamp_min(0).cpu().numpy()
                    feat = model.rssm.get_feat(z,h)
                    rr = model.reward(feat).mode().reshape(-1,1)
                    cc = model.cont(feat).mean.reshape(-1,1)
                    channels = [rr,cc]
                    if decoder:
                        state = model.decoder(z[:,None],h[:,None])['state'].mode()[:,0]
                        channels.append(state[:,:19])
                    result.append(torch.cat(channels,-1).cpu().numpy())
                sequence = np.stack(result,1)
                assert np.isfinite(sequence).all()
                if variant < 9:
                    predicted[variant,start:stop] = sequence
                else:
                    exact_self_check &= np.array_equal(sequence,predicted[0,start:stop])
            print(a.label,'counterfactual origins',stop,'/',n,flush=True)
    assert exact_self_check and exact_future_only_check
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == weight_hash
    np.savez_compressed(a.output/'counterfactuals.npz',predicted=predicted,posterior_kl=posterior_kl,
                        prior_js=prior_js,initial_latent_change=initial_latent_change,sigma=sigma)
    provenance = dict(label=a.label,checkpoint=str(checkpoint.resolve()),checkpoint_step=step,
        checkpoint_sha256=weight_hash,source=str(a.source.resolve()),corpus=str(a.corpus.resolve()),
        samples=n,horizon=H,conditioning_window=K,context_sigma=sigma.tolist(),native_decoder=decoder,
        channels=['reward','continuation']+([f'native_state_{i}' for i in range(19)] if decoder else []),
        physical_mapping={'native_state_3':'speed_kmh = x * 51 - 1',
                          'native_state_7':'unsigned clipped yaw-rate observation, normalized [0,1]'},
        method='DALI-inspired +1 sigma coordinate interventions. Shared causal pre-window latent state; identical real observation/action window; original posterior recomputed with original or perturbed context. Then 50 zero-action prior steps with common Gumbel noise. Original weights and architecture unchanged.',
        adaptation='The trained Shallow model has no future-context input to its transition/prior. Interventions therefore enter its native posterior during the last five causal observations. Future-only injection is separately verified to be an exact null. This is not a bitwise reproduction of the DALI code.',
        rng='One paired latent draw per origin; fixed seed per batch, identical stochastic noise in every branch. These are not independent training seeds.',
        no_future_truth_used=True,new_decoder_fitted=False,physical_parameter_identification_tested=False,
        exact_self_intervention=exact_self_check,exact_future_only_null=exact_future_only_check,
        limitation='Predictive sensitivity does not establish physically correct mass/friction counterfactuals, improved prediction accuracy, or a causal explanation of driving gains. Native reward/continuation are used for decoder-free R2; no auxiliary dynamics are presented as RSSM predictions.',
        reference='https://arxiv.org/html/2508.20294v3')
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(status='completed')))
    print('COMPLETED',a.output,flush=True)


if __name__ == '__main__':
    main()
