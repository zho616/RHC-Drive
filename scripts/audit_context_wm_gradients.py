"""No-update gradient audit on cached T-intersection episodes, using current code."""
import json, sys, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import gymnasium as gym
from tensordict import TensorDict
from omegaconf import OmegaConf
from dreamer import Dreamer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('logdir/tintersection_mass_friction_500k_20261021'))
    p.add_argument('--batches',type=int,default=6)
    a=p.parse_args(); out=a.root/'context_wm_gradient_audit_20261022';out.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    corpus=a.root/'world_model_common_corpus'; entries=json.loads((corpus/'episodes.json').read_text())
    arrays=[dict(np.load(corpus/'episodes'/e['file'])) for e in entries]
    keys=[k for k in arrays[0] if k not in ('executed_action','speed','wheel_degrees')]
    for i,d in enumerate(arrays):
        d['action']=np.concatenate([np.zeros((1,2),np.float32),d['executed_action'][:-1]])
        d['ctx_episode_uid']=np.tile([0,i+1],(len(d['state']),1))
        for k in keys:
            if d[k].ndim==1:d[k]=d[k][:,None]
    spaces=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=arrays[0][k].shape[1:],dtype=np.float32) for k in keys if k!='reward'})
    run=a.root/'training/r2dreamer_dali_shallow';cfg=OmegaConf.load(run/'.hydra/config.yaml');cfg.device='cuda:0';OmegaConf.resolve(cfg)
    cfg.model.compile=False;cfg.model.context.wm_grad_scale=1.
    model=Dreamer(cfg.model,spaces,gym.spaces.Box(-1,1,(2,),np.float32)).cuda()
    payload=torch.load(run/'best.pt',map_location='cuda:0',weights_only=False)
    model.load_state_dict(payload['agent_state_dict'],strict=True);step=payload['step'];del payload
    model.clone_and_freeze();model.eval()
    ep=list(model.context_encoder.parameters())
    wp=[p for n,p in model._named_params.items() if not n.startswith(('context_encoder.','context_dynamics.'))]
    dp=list(model.context_dynamics.parameters())
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten() for g,p in zip(gs,ps)])
    rows=[]; original=model._context_wm_surrogate
    def inspect(losses,context,leaf):
        vectors={};row={'losses':{k:float(v.detach()) for k,v in losses.items()}}
        for key in ('context_aux','context_swap'):
            objective=losses[key]*model._loss_scales[key]
            vectors[key]=flat(torch.autograd.grad(objective,ep,retain_graph=True,allow_unused=True),ep)
            gs=torch.autograd.grad(objective,wp,retain_graph=True,allow_unused=True)
            assert all(g is None or torch.count_nonzero(g)==0 for g in gs),key
        for key in model.CONTEXT_WM_LOSSES:
            g,=torch.autograd.grad(losses[key]*model._loss_scales[key],leaf,retain_graph=True)
            vectors[key]=flat(torch.autograd.grad(context,ep,grad_outputs=g.detach(),retain_graph=True,allow_unused=True),ep)
        vectors['aux_combined']=vectors['context_aux']+vectors['context_swap']
        vectors['wm_combined']=sum(vectors[k] for k in model.CONTEXT_WM_LOSSES)
        surrogate,metrics=original(losses,context,leaf)
        gs=torch.autograd.grad(surrogate,wp+dp,retain_graph=True,allow_unused=True)
        assert all(g is None for g in gs)
        routed=flat(torch.autograd.grad(surrogate,ep,retain_graph=True,allow_unused=True),ep)
        torch.testing.assert_close(routed,vectors['wm_combined'],rtol=2e-3,atol=2e-5)
        row['norms']={k:float(v.norm()) for k,v in vectors.items()}
        x,y=vectors['wm_combined'],vectors['aux_combined']
        row['ratio']=float(x.norm()/y.norm().clamp_min(1e-12))
        row['cosine']=float(torch.nn.functional.cosine_similarity(x,y,dim=0))
        row['isolation_pass']=True;rows.append(row)
        return surrogate,metrics
    model._context_wm_surrogate=inspect
    rng=np.random.default_rng(20261022)
    eligible=[i for i,d in enumerate(arrays) if len(d['state'])>=64]
    pairs=[(i,i+1) for i in range(0,len(arrays),2) if i in eligible and i+1 in eligible]
    for batch in range(a.batches):
        ids=(rng.choice(eligible,16,replace=False).tolist() if batch%2==0 else [i for j in rng.choice(len(pairs),8,replace=False) for i in pairs[j]])
        # Prefixes preserve true reset and avoid guessed replay recurrent states.
        td=TensorDict({k:torch.as_tensor(np.stack([arrays[i][k][:64] for i in ids]),device='cuda:0') for k in keys+['action']},batch_size=(16,64))
        model.zero_grad(set_to_none=True);torch.manual_seed(1000+batch)
        _,metrics=model._cal_grad(td,model.rssm.initial(16))
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        rows[-1].update(batch=batch,sampling='uniform' if batch%2==0 else 'paired',episodes=ids,swap_pairs=float(metrics['context/swap_pairs']))
        print(json.dumps(rows[-1]),flush=True)
        del metrics,td
    model._context_wm_surrogate=original
    model._scaler=torch.amp.GradScaler(init_scale=128.)
    model.zero_grad(set_to_none=True);torch.manual_seed(20261022)
    td=TensorDict({k:torch.as_tensor(np.stack([arrays[i][k][:64] for i in ids]),device='cuda:0') for k in keys+['action']},batch_size=(16,64))
    with torch.autocast('cuda',dtype=torch.float16):
        model._cal_grad(td,model.rssm.initial(16))
    amp_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert amp_finite, 'AMP nonfinite gradients'
    report={'amp_smoke':{'finite_gradients':amp_finite,'grad_scale':128},'checkpoint':str((run/'best.pt').resolve()),'step':step,'precision':'FP32; training uses AMP',
      'protocol':'6 no-update batches by default, B16 T64 true episode prefixes; uniform/paired sampling; no optimizer steps. Norms measured before AGC/LaProp.',
      'limitations':'Early-episode prefixes, held-out cached trajectories, one checkpoint; paired batches oversample swap availability versus replay.',
      'loss_scales':dict(model._loss_scales),'rows':rows}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
