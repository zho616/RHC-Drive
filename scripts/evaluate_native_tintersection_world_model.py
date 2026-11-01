"""Native frozen-model KL and posterior state reconstruction on a common corpus.

No decoder is added or fitted. Original source/config/checkpoint are used.
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
    p.add_argument('--checkpoint', default='best.pt', choices=['best.pt', 'latest.pt'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=10)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    import distributions as dists
    from dreamer import Dreamer
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cfg = OmegaConf.load(a.model/'.hydra/config.yaml')
    cfg.device = a.device
    OmegaConf.resolve(cfg)
    cfg.model.compile = False
    episodes = json.loads((a.corpus/'episodes.json').read_text())
    data = [dict(np.load(a.corpus/'episodes'/e['file'])) for e in episodes]
    dim = data[0]['state'].shape[-1]
    for d in data:
        assert d['state'].shape[-1] == dim
        assert np.isfinite(d['state']).all() and d['state'].min() >= 0 and d['state'].max() <= 1
        assert d['ctx_ego_state'].shape[-1] == 4 and d['ctx_history'].shape[-2:] == (5, 6)
    excluded = {'executed_action', 'speed', 'wheel_degrees', 'reward'}
    spaces = {k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape[1:], dtype=np.float32)
              for k, v in data[0].items() if k not in excluded}
    model = Dreamer(cfg.model, gym.spaces.Dict(spaces),
                    gym.spaces.Box(-1, 1, (2,), np.float32)).to(a.device)
    checkpoint = a.model/a.checkpoint
    payload = torch.load(checkpoint, map_location=a.device, weights_only=False)
    model.load_state_dict(payload['agent_state_dict'], strict=True)
    step = int(payload['step'])
    del payload
    model.clone_and_freeze()
    model.eval()
    context_on = model.context_enabled
    decoder_on = hasattr(model, 'decoder')
    if context_on:
        assert cfg.model.context.injection == 'dali_shallow'
    assert decoder_on == (str(cfg.model.rep_loss) == 'dreamer')
    n, tmax = len(data), max(len(d['state']) for d in data)
    keys = ['kl_prior_post_raw', 'kl_post_prior_raw', 'kl_post_prior_free',
            'kl_prior_post_unimix', 'kl_post_prior_unimix', 'prior_entropy', 'post_entropy']
    if decoder_on:
        keys += ['state_native_recon_loss', 'state_mae', 'state_mse', 'state_psnr_frame']
    values = {k: np.full((a.draws, n, tmax), np.nan, np.float32) for k in keys}
    valid = np.zeros((n, tmax), bool)
    for i, d in enumerate(data):
        count = len(d['state'])
        valid[i, :count] = ((np.arange(count) >= 20) & d['ctx_mask'].all(-1)
                            & ~d['is_last'] & ~d['ctx_collision'])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    corpus_hash = hashlib.sha256((a.corpus/'episodes.json').read_bytes()).hexdigest()
    tensor = lambda x: torch.as_tensor(x, device=a.device)
    with torch.inference_mode():
        for start in range(0, n, a.batch_size):
            ids = list(range(start, min(n, start+a.batch_size)))
            length = max(len(data[i]['state']) for i in ids)
            batch = {}
            for key in ['state', 'is_first', 'ctx_history', 'ctx_mask']:
                arr = np.zeros((len(ids), length, *data[0][key].shape[1:]), data[0][key].dtype)
                for b, i in enumerate(ids):
                    arr[b, :len(data[i][key])] = data[i][key]
                batch[key] = tensor(arr)
            embedding = model.encoder(TensorDict(batch, batch_size=[len(ids), length]))
            context = model.context_encoder(batch['ctx_history'], batch['ctx_mask']) if context_on else None
            prev = torch.zeros(len(ids), length, 2, device=a.device)
            for b, i in enumerate(ids):
                prev[b, 1:len(data[i]['state'])] = tensor(data[i]['executed_action'][:-1])
            for draw in range(a.draws):
                torch.manual_seed(20261025 + draw*100000 + start)
                z, h = model.rssm.initial(len(ids))
                collected = {k: [] for k in keys}
                for t in range(length):
                    extra = (context[:, t],) if context_on else ()
                    z, h, q = model.rssm.obs_step(z, h, prev[:, t], embedding[:, t],
                                                 batch['is_first'][:, t], *extra)
                    # Shallow's prior head is unconditioned; access logits without
                    # drawing an unused prior sample or changing posterior RNG.
                    prior = model.rssm._img_net(h)
                    qp = dists.kl(q.float(), prior.float()).sum(-1)
                    pq = dists.kl(prior.float(), q.float()).sum(-1)
                    qdist, pdist = model.rssm.get_dist(q), model.rssm.get_dist(prior)
                    row = dict(kl_prior_post_raw=pq, kl_post_prior_raw=qp,
                               kl_post_prior_free=qp.clamp_min(model.kl_free),
                               kl_prior_post_unimix=torch.distributions.kl_divergence(pdist, qdist),
                               kl_post_prior_unimix=torch.distributions.kl_divergence(qdist, pdist),
                               prior_entropy=pdist.entropy(), post_entropy=qdist.entropy())
                    if decoder_on:
                        c = context[:, t:t+1] if context_on else None
                        hd = model._prediction_feat(h[:, None], c)
                        distribution = model.decoder(z[:, None], hd)['state']
                        target = batch['state'][:, t:t+1]
                        prediction = distribution.mode()
                        error = prediction-target
                        mse = error.square().mean(-1).squeeze(1)
                        row.update(state_native_recon_loss=-distribution.log_prob(target).squeeze(1),
                                   state_mae=error.abs().mean(-1).squeeze(1), state_mse=mse,
                                   state_psnr_frame=-10*torch.log10(mse.clamp_min(1e-12)))
                    for key, value in row.items():
                        collected[key].append(value.float())
                for key, rows in collected.items():
                    arr = torch.stack(rows, dim=1).cpu().numpy()
                    for b, i in enumerate(ids):
                        count = len(data[i]['state'])
                        values[key][draw, i, :count] = arr[b, :count]
            print(a.label, 'filtered', start+len(ids), '/', n, flush=True)
    for key, arr in values.items():
        assert np.isfinite(arr[:, valid]).all(), key
    np.savez_compressed(a.output/'native_metrics.npz', **values, valid=valid)
    records = []
    for i, episode in enumerate(episodes):
        record = dict(episode=i, **episode, valid_frames=int(valid[i].sum()))
        if valid[i].any():
            record['metrics'] = {k: float(v[:, i, valid[i]].mean(dtype=np.float64)) for k,v in values.items()}
            if decoder_on:
                record['metrics']['state_psnr_pooled'] = float(-10*np.log10(record['metrics']['state_mse']))
        else:
            record['metrics'] = None
        records.append(record)
    provenance = dict(label=a.label, checkpoint=str(checkpoint.resolve()), checkpoint_step=step,
        checkpoint_selection=a.checkpoint,
        checkpoint_sha256=checkpoint_hash, source=str(a.source.resolve()),
        corpus=str(a.corpus.resolve()), corpus_manifest_sha256=corpus_hash,
        corpus_provenance=json.loads((a.corpus/'provenance.json').read_text()),
        state_dim=dim, state_range=[0,1], psnr_peak=1., prediction_clipping=False,
        trained_decoder=decoder_on, new_decoder_fitted=False, representation_loss=str(cfg.model.rep_loss),
        context_injection=str(cfg.model.context.injection) if context_on else None,
        latent_factors=int(model.rssm._stoch), categories=int(model.rssm._discrete),
        kl_free=float(model.kl_free), unimix=float(model.rssm._unimix_ratio),
        draws=a.draws, latent_seeds=[20261025+d*100000 for d in range(a.draws)],
        seed_rule='base + batch start index; identical episode batching for every model',
        device=a.device, precision='float32, highest matmul precision',
        eligibility='All real frames t>=20 with full causal context history; exclude is_last/collision; identical across methods.',
        kl_definition='Sum over all categorical latent factors, nats, no free-nats clipping for primary raw KL. Both directions reported. Native training uses un-mixed posterior||prior logits; sampling uses unimix, reported separately.',
        reconstruction_definition='Native trained decoder mode on posterior states. Raw normalized observation MAE/MSE, state PSNR with peak=1. No clipping. Native reconstruction loss is summed symlog squared error, not Gaussian NLL or BEV loss.',
        reconstruction_missing_reason=None if decoder_on else 'Decoder-free R2Dreamer; no native observation reconstruction head.',
        evaluation_scope='Observation-conditioned posterior reconstruction and one-step prior/posterior alignment; not open-loop future prediction accuracy.')
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (a.output/'episodes.json').write_text(json.dumps(records, indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(status='completed',label=a.label)))
    print('COMPLETED', a.output, flush=True)


if __name__ == '__main__':
    main()
