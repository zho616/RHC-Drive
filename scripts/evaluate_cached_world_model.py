"""Compare frozen world models on a common cached trajectory corpus (no simulator).

Run separately for each frozen source to avoid Python import/version contamination.
The posterior uses the observation prefix; forecasts use only recorded future actions.
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
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--draws', type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from dreamer import Dreamer
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cfg = OmegaConf.load(args.model / '.hydra/config.yaml')
    cfg.device = args.device
    OmegaConf.resolve(cfg)
    cfg.model.compile = False
    context_on = bool(getattr(cfg.model, 'context', None) and cfg.model.context.enabled)
    if context_on:
        from context import refresh_context, predict_next_state, canonical_action
    episodes = json.loads((args.corpus / 'episodes.json').read_text())
    origins = json.loads((args.corpus / 'origins.json').read_text())
    data = [dict(np.load(args.corpus / 'episodes' / e['file'])) for e in episodes]
    for d in data:
        d['reward'] = sum(d['log_reward_' + k] for k in ['progress', 'speed', 'lateral', 'terminal']).astype(np.float32)
        old = d['ctx_history']
        original_state_dim = d['ctx_ego_state'].shape[-1]
        assert original_state_dim in (2, 4)
        state = (np.column_stack((d['ctx_ego_state'], d['speed'], -np.deg2rad(d['wheel_degrees'])))
                 if original_state_dim == 2 else d['ctx_ego_state']).astype(np.float32)
        np.testing.assert_allclose(state[:, 2], d['speed'], atol=1e-5)
        np.testing.assert_allclose(state[:, 3], -np.deg2rad(d['wheel_degrees']), atol=1e-5)
        hist = np.zeros((len(state), 5, 6), np.float32)
        for t in range(len(state)):
            count = int(d['ctx_mask'][t].sum())
            if count:
                sl = slice(t-count+1, t+1)
                np.testing.assert_allclose(old[t, -count:, :original_state_dim], state[sl, :original_state_dim], atol=1e-6)
                np.testing.assert_allclose(old[t, -count:, original_state_dim:], d['ctx_prev_action'][sl], atol=1e-6)
                hist[t, -count:] = np.column_stack((state[sl], d['ctx_prev_action'][sl]))
        d['ctx_ego_state'], d['ctx_history'] = state, hist
        d['ctx_episode_uid'] = np.zeros((len(state), 2), np.int64)
    excluded = {'executed_action', 'speed', 'wheel_degrees', 'reward'}
    spaces = {k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape[1:], dtype=np.float32)
              for k, v in data[0].items() if k not in excluded}
    model = Dreamer(cfg.model, gym.spaces.Dict(spaces), gym.spaces.Box(-1, 1, (2,), np.float32)).to(args.device)
    checkpoint = args.model / 'best.pt'
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload['agent_state_dict'], strict=True)
    step = payload['step']; del payload
    model.clone_and_freeze(); model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'checkpoint': str(checkpoint.resolve()), 'step': step,
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'source': str(args.source.resolve()),
        'corpus': str(args.corpus.resolve()), 'origins_sha256': hashlib.sha256((args.corpus/'origins.json').read_bytes()).hexdigest(),
        'draws': args.draws, 'device': args.device, 'precision': 'float32, highest matmul precision',
        'protocol': 'Common natural trajectories, action-conditioned open loop, dynamic predicted context, no future observations/contexts. No predicted-continuation gating; post-episode targets masked.',
        'corpus_limits': 'Shared cached corpus; one trained checkpoint per method. See corpus provenance for map/chassis sampling.',
        'corpus_provenance': json.loads((args.corpus/'provenance.json').read_text()) if (args.corpus/'provenance.json').exists() else None}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    device = args.device
    def tensor(x): return torch.as_tensor(x, device=device)
    n, H = len(origins), 10
    z_all = torch.empty(n, model.rssm._stoch, model.rssm._discrete, device=device)
    h_all = torch.empty(n, model.rssm._deter, device=device)
    with torch.inference_mode():
        torch.manual_seed(314159)
        for start in range(0, len(episodes), 10):
            chunk = list(range(start, min(start+10, len(episodes))))
            maxlen = max(len(data[i]['state']) for i in chunk)
            td = {}
            for key in ['state', 'is_first', 'ctx_history', 'ctx_mask']:
                arr = np.zeros((len(chunk), maxlen, *data[0][key].shape[1:]), dtype=data[0][key].dtype)
                for b, ei in enumerate(chunk): arr[b, :len(data[ei][key])] = data[ei][key]
                td[key] = tensor(arr)
            emb = model.encoder(TensorDict(td, batch_size=[len(chunk), maxlen]))
            cc = model.context_encoder(td['ctx_history'], td['ctx_mask']) if context_on else None
            prev = torch.zeros(len(chunk), maxlen, 2, device=device)
            for b, ei in enumerate(chunk): prev[b, 1:len(data[ei]['state'])] = tensor(data[ei]['executed_action'][:-1])
            z, h = model.rssm.initial(len(chunk)); lookup = {}
            for i, o in enumerate(origins):
                if o['episode'] in chunk: lookup.setdefault(o['t'], []).append((i, chunk.index(o['episode'])))
            for t in range(maxlen):
                extra = (cc[:, t],) if context_on else ()
                z, h, _ = model.rssm.obs_step(z, h, prev[:, t], emb[:, t], td['is_first'][:, t], *extra)
                for i, b in lookup.get(t, []): z_all[i], h_all[i] = z[b], h[b]
            print('filtered', start+len(chunk), '/', len(episodes), flush=True)
        acts = np.zeros((n, H, 2), np.float32)
        reward = np.zeros((n, H, 1), np.float32); cont = np.zeros_like(reward)
        truth = np.zeros((n, H, 4), np.float32); valid = np.zeros((n, H), bool); physical = np.zeros_like(valid)
        for i, o in enumerate(origins):
            d = data[o['episode']]; t = o['t']; length = min(H, episodes[o['episode']]['length']-t)
            acts[i, :length] = d['executed_action'][t:t+length]
            truth[i, :length] = d['ctx_ego_state'][t+1:t+length+1]
            reward[i, :length, 0] = d['reward'][t+1:t+length+1]
            cont[i, :length, 0] = 1-d['is_terminal'][t+1:t+length+1]
            valid[i, :length] = True
            physical[i, :length] = ~(d['is_last'][t+1:t+length+1] | d['ctx_collision'][t+1:t+length+1])
            np.testing.assert_allclose(d['ctx_prev_action'][t+1:t+length+1], np.stack((acts[i, :length, 1], -acts[i, :length, 0]), -1), atol=1e-6)
        initial = np.stack([data[o['episode']]['ctx_ego_state'][o['t']] for o in origins])
        histories = np.stack([data[o['episode']]['ctx_history'][o['t']] for o in origins])
        masks = np.stack([data[o['episode']]['ctx_mask'][o['t']] for o in origins])
        errors = {key: np.zeros((args.draws, n, H), np.float32) for key in ['reward_mae', 'reward_nll', 'continue_brier', 'continue_nll']}
        for start in range(0, n, 128):
            sl = slice(start, min(n, start+128)); count = sl.stop-sl.start
            for draw in range(args.draws):
                torch.manual_seed(900000+start+draw*100000)
                z, h = z_all[sl].clone(), h_all[sl].clone()
                if context_on:
                    state, hist, mask = tensor(initial[sl]), tensor(histories[sl]), tensor(masks[sl])
                    c = model.context_encoder(hist, mask)
                for k in range(H):
                    if context_on:
                        state, c, hist, mask = refresh_context(model.context_encoder, model.context_dynamics, state, tensor(acts[sl, k]), c, hist, mask, torch.ones(count, dtype=torch.bool, device=device))
                    z, h = model.rssm.img_step(z, h, tensor(acts[sl, k]), *((c,) if context_on else ()))
                    feat = model.rssm.get_feat(z, h)
                    if hasattr(model, '_prediction_feat'): feat = model._prediction_feat(feat, c if context_on else None)
                    rw, cn = model.reward(feat), model.cont(feat)
                    rt, ct = tensor(reward[sl, k]), tensor(cont[sl, k])
                    values = {'reward_mae': (rw.mode()-rt).abs(), 'reward_nll': -rw.log_prob(rt),
                              'continue_brier': (cn.mean-ct).square(), 'continue_nll': -cn.log_prob(ct)}
                    for key, value in values.items(): errors[key][draw, sl, k] = value.reshape(-1).cpu().numpy()
            print('forecast', sl.stop, '/', n, flush=True)
        if context_on:
            state = tensor(initial); c = model.context_encoder(tensor(histories), tensor(masks)); aux = []
            for k in range(H):
                state = predict_next_state(model.context_dynamics, c, state, canonical_action(tensor(acts[:, k])))
                aux.append((state-tensor(truth[:, k])).abs().cpu().numpy())
            errors['auxiliary_mae'] = np.stack(aux, 1)
    np.savez_compressed(args.output/'errors.npz', **errors, valid=valid, physical_valid=physical)
    summary = {}
    for split, select in [('all', np.ones(n, bool)), ('seen_types', np.array([o['model'] in ['default','m','l'] for o in origins])), ('unseen_types', np.array([o['model'] in ['s','xl'] for o in origins]))]:
        summary[split] = {key: {str(k+1): float(value.mean(0)[select & valid[:, k], k].mean()) for k in [0,4,9]} for key, value in errors.items() if key != 'auxiliary_mae'}
        if context_on:
            summary[split]['auxiliary_mae'] = {str(k+1): errors['auxiliary_mae'][select & physical[:, k], k].mean(0).tolist() for k in [0,4,9]}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('COMPLETED', args.output, flush=True)


if __name__ == '__main__':
    main()
