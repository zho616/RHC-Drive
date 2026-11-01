"""Group-held-out trajectory discrimination and paired intervention summaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np


def save_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deps', type=Path, default=Path('/tmp/r2dreamer-counterfactual-deps'))
    a = p.parse_args()
    sys.path.insert(0,str(a.deps))
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    warnings.filterwarnings('ignore',category=FutureWarning,module='sklearn.svm')
    threadpool_limits(limits=1)
    rows,curve_rows,example_rows = [],[],[]
    report = {}
    for label in ['r2dreamer_context','dreamerv3_context']:
        folder = a.root/label
        origins = json.loads((folder/'origins.json').read_text())
        provenance = json.loads((folder/'provenance.json').read_text())
        d = dict(np.load(folder/'counterfactuals.npz'))
        pred,qkl,js = d['predicted'],d['posterior_kl'],d['prior_js']
        n,H = pred.shape[1:3]
        groups = np.array([o['map_id'] for o in origins])
        map_ids = np.unique(groups)
        rng = np.random.default_rng(20261025)
        boot = rng.choice(len(map_ids),(1000,len(map_ids)),replace=True)
        def interval(values):
            bymap = np.array([values[groups==g].mean(axis=0) for g in map_ids])
            resampled = bymap[boot].mean(1)
            return bymap.mean(0),np.quantile(resampled,[.025,.975],axis=0)
        auc_file = folder/'auc_results.json'
        if auc_file.exists():
            auc_results = json.loads(auc_file.read_text())
        else:
            auc_results = []
            # All 50 timesteps, native reward/continue. For DreamerV3, add
            # native ego/navigation state at 5-step intervals (no fitted decoder).
            def features(x):
                xx = [x[:,:,:2].reshape(n,-1)]
                if x.shape[-1] > 2:
                    xx.append(x[:,4::5,2:].reshape(n,-1))
                return np.concatenate(xx,-1)
            x0 = features(pred[0])
            y = np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]
            all_groups = np.r_[groups,groups]
            oof_all = np.empty((8,2*n),np.float64)
            for dim in range(8):
                X = np.concatenate([x0,features(pred[dim+1])])
                probs = np.zeros(2*n)
                fold_aucs = []
                def fit_fold(fold):
                    test = np.isin(all_groups,map_ids[fold::5]);train = ~test
                    assert not set(all_groups[train]) & set(all_groups[test])
                    scale = StandardScaler().fit(X[train])
                    xt,xv = scale.transform(X[train]),scale.transform(X[test])
                    classifiers = [
                        SVC(C=30,kernel='rbf',probability=True,random_state=20261025+dim),
                        MLPClassifier(hidden_layer_sizes=(64,32),alpha=.001,max_iter=200,
                                      early_stopping=False,random_state=20261025+dim),
                        AdaBoostClassifier(n_estimators=100,random_state=20261025+dim)]
                    scores = []
                    for clf in classifiers:
                        clf.fit(xt,y[train]);scores.append(clf.predict_proba(xv)[:,1])
                    scores = np.mean(scores,axis=0)
                    return test,scores,float(roc_auc_score(y[test],scores))
                with ThreadPoolExecutor(max_workers=5) as executor:
                    for test,scores,fold_auc in executor.map(fit_fold,range(5)):
                        probs[test] = scores
                        fold_aucs.append(fold_auc)
                # Bootstrap complete maps, retaining pairs and correlated windows.
                boot_aucs = []
                for selected in boot[:500]:
                    indices = np.concatenate([np.flatnonzero(all_groups==map_ids[g]) for g in selected])
                    boot_aucs.append(roc_auc_score(y[indices],probs[indices]))
                result = dict(dimension=dim+1,auc=float(roc_auc_score(y,probs)),
                              ci95=np.quantile(boot_aucs,[.025,.975]).tolist(),fold_aucs=fold_aucs)
                auc_results.append(result);oof_all[dim] = probs
                print(label,'dimension',dim+1,'AUC',result['auc'],flush=True)
            np.savez_compressed(folder/'auc_oof.npz',probabilities=oof_all,labels=y,map_ids=all_groups)
            auc_file.write_text(json.dumps(auc_results,indent=2))
        for dim in range(8):
            changes = np.abs(pred[dim+1]-pred[0])
            metrics = dict(posterior_kl=qkl[dim],prior_js=js[dim].mean(1),
                           reward_abs_change=changes[:,:,0].mean(1),
                           continuation_abs_change=changes[:,:,1].mean(1),
                           initial_latent_change=d['initial_latent_change'][dim].astype(float))
            if pred.shape[-1]>2:
                metrics['speed_abs_change_kmh'] = changes[:,:,2+3].mean(1)*51
                metrics['yaw_observation_abs_change'] = changes[:,:,2+7].mean(1)
            row = dict(method=label,sigma=float(d['sigma'][dim]),**auc_results[dim])
            row.pop('fold_aucs')
            lo,hi = row.pop('ci95');row['auc_ci_low'],row['auc_ci_high'] = lo,hi
            for metric,value in metrics.items():
                mean,ci = interval(value)
                row[metric]=float(mean);row[metric+'_ci_low']=float(ci[0]);row[metric+'_ci_high']=float(ci[1])
            rows.append(row)
            for metric,values in [('prior_js',js[dim]),('reward_abs_change',changes[:,:,0]),
                                  ('continuation_abs_change',changes[:,:,1])]:
                mean,ci = interval(values)
                for t in range(H):
                    curve_rows.append(dict(method=label,dimension=dim+1,metric=metric,horizon=t+1,
                                           mean=float(mean[t]),ci_low=float(ci[0,t]),ci_high=float(ci[1,t])))
        # A representative, not the maximum-effect, example. Select the median
        # nonzero response of the strongest posterior-KL dimension.
        top = int(np.argmax(qkl.mean(1)))
        channel = 2+3 if pred.shape[-1]>2 else 0
        scale = 51. if channel!=0 else 1.
        effect = np.abs(pred[top+1,:,:,channel]-pred[0,:,:,channel]).mean(1)*scale
        eligible = np.flatnonzero(effect>1e-7)
        oi = int(eligible[np.argsort(effect[eligible])[len(eligible)//2]])
        # Also preserve a clearly visible upper-quartile example, explicitly labeled.
        ex_indices = [oi,int(eligible[np.argsort(effect[eligible])[3*len(eligible)//4]])]
        for example_kind,idx in zip(['median_nonzero','upper_quartile_nonzero'],ex_indices):
            for variant in [0,top+1]:
                for t in range(H):
                    example_rows.append(dict(method=label,example=example_kind,origin=idx,dimension=top+1,
                        episode=origins[idx]['episode'],map_id=origins[idx]['map_id'],t=origins[idx]['t'],
                        horizon=t+1,branch='Original' if variant==0 else '+1 SD context',
                        reward=float(pred[variant,idx,t,0]),continuation=float(pred[variant,idx,t,1]),
                        speed_kmh=float(pred[variant,idx,t,5]*51-1) if pred.shape[-1]>2 else '',
                        yaw_observation=float(pred[variant,idx,t,9]) if pred.shape[-1]>2 else ''))
        report[label] = dict(provenance=provenance,dimensions=rows[-8:],
            strongest_posterior_dimension=top+1,representative_origins=ex_indices,
            classifier='SVM RBF C=30 + MLP(64,32) + AdaBoost(100), mean probabilities; train-fold scaling. Five map-held-out folds; original/perturbed pairs and all windows from a map stay in the same fold. MLP has no internal random validation split.',
            auc_scope='Distinguish original and perturbed native predicted sequences, not future accuracy. Native reward/continue used by both; DreamerV3 adds 19 ego/navigation decoder channels sampled every five steps. AUC is not comparable across model families because their native outputs differ.',
            intervals='95% bootstrap over 10 map blocks, 500 AUC resamples and 1000 paired-sensitivity resamples; one frozen seed-0 model, not uncertainty over training seeds.')
    # Missing native-state metrics are explicit empty cells for R2.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with(a.root/'dimension_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    save_csv(a.root/'sensitivity_curves.csv',curve_rows)
    save_csv(a.root/'representative_predictions.csv',example_rows)
    (a.root/'summary.json').write_text(json.dumps(report,indent=2))
    print('COMPLETED',a.root,flush=True)


if __name__=='__main__':
    main()
