"""Build reproducible derived tables and submission documentation, never train models."""
import csv,json,hashlib,sys
from pathlib import Path
from statistics import mean,stdev
import torch,numpy as np
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from analysis_code.final_acceptance import DEFAULT as O,verify_outputs
from refinement_v3.common import read_csv,write_csv,write_json
from refinement_v3.config import contrasts,sign_test
from refinement_v3.calibration import calibration_metrics,reliability_rows
from refinement_v3.plots import reliability

def main():
    global O
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=O);a=parser.parse_args();O=a.out
    verify_outputs(O)
    if json.loads((O/'completion.json').read_text(encoding='utf8')).get('finalized'):
        print('Finalized outputs already verified; no overwrite')
        return
    rows=read_csv(O/'per_seed.csv');cal=read_csv(O/'calibration_per_seed.csv')
    lookup={(('stage2:' if r['stage']=='2' else '')+r['run'],int(r['seed'])):float(r['test_accuracy']) for r in rows}
    effects=[]
    for name,weights in contrasts().items():
        ds=[100*sum(w*lookup[k,s] for k,w in weights.items()) for s in [20261105,20261106,20261107]]
        p,n=sign_test(ds)
        effects.append(dict(contrast=name,seed_20261105_pp=ds[0],seed_20261106_pp=ds[1],seed_20261107_pp=ds[2],
            mean_pp=mean(ds),sample_sd_pp=stdev(ds),two_sided_sign_p=p,n=n,
            inference='descriptive; 3 seeds; minimum possible two-sided sign p=0.25'))
    write_csv(O/'contrasts.csv',effects)
    grouped=[]
    for stage,name in sorted({(r['stage'],r['run']) for r in cal}):
        block=[r for r in cal if r['stage']==stage and r['run']==name];g={'stage':stage,'run':name,'n_seeds':3}
        for key in ['temperature','accuracy','ece_raw','ece_cal','nll_raw','nll_cal','brier_raw','brier_cal']:
            v=[float(r[key]) for r in block];g[key+'_mean']=mean(v);g[key+'_sd']=stdev(v)
        grouped.append(g)
        if name not in ['baseline_both','se_both','cbam_both','ls_both','convstem_plain','convstem_both','mixup_both','cutmix_both']:continue
        record=next(r for r in block if r['seed']=='20261105')
        folder=O/f'stage{stage}'/name/'20261105';data=np.load(folder/'logits.npz')
        for kind,t in [('raw',1.),('calibrated',float(record['temperature']))]:
            m=calibration_metrics(torch.tensor(data['test']),torch.tensor(data['test_labels']),t)
            points={k:[b[k] for b in m['bins'] if b['count']] for k in ['confidence','accuracy']}
            reliability(points,f'{name}: validation-fitted temperature, test ({kind})',folder/f'reliability_{kind}.png')
            write_csv(folder/f'reliability_{kind}.csv',reliability_rows(m,name,'test_'+kind))
    write_csv(O/'calibration_grouped.csv',grouped)
    convergence=[]
    for r in rows:
        ck=P/r['checkpoint'];hist=read_csv(ck.parent/'history.csv');best=None;stale=0
        for h in hist:
            metric=(float(h['val_accuracy']),-float(h['val_loss']))
            if best is None or metric>best:best=metric;stale=0
            else:stale+=1
        assert r['stop_reason']!='early_stopping' or stale>=12
        for h in hist:
            e=int(h['epoch']);expected=.00005+(.002-.00005)*(1+np.cos(np.pi*min(e-1,60)/60))/2
            assert abs(float(h['lr'])-expected)<1e-12
        selected=hist[int(r['best_epoch'])-1]
        convergence.append(dict(stage=r['stage'],run=r['run'],seed=r['seed'],epochs=r['epochs'],best_epoch=r['best_epoch'],
            stale_epochs_at_end=stale,train_accuracy_at_selection=selected['train_accuracy'],
            val_accuracy_at_selection=selected['val_accuracy'],train_loss_at_selection=selected['train_loss'],
            val_loss_at_selection=selected['val_loss'],last_train_accuracy=hist[-1]['train_accuracy'],
            last_val_accuracy=hist[-1]['val_accuracy'],last_val_loss=hist[-1]['val_loss']))
    write_csv(O/'convergence_audit.csv',convergence)
    # New artifacts added above extend the derivative manifest, not the immutable formal hashes.
    from analysis_code.canonical import sha
    write_json(O/'completion.json',{'formal_runs':39,'finalized':True,'files':{str(f.relative_to(O)):sha(f) for f in O.rglob('*')
               if f.is_file() and f.name!='completion.json' and 'robustness' not in f.relative_to(O).parts}})

if __name__=='__main__':main()
