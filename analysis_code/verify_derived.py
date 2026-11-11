"""Validate cached logits, class statistics, calibration invariance and robustness aggregation."""
import sys
from pathlib import Path
from statistics import mean,stdev
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from refinement_v3.common import read_csv,read_json
from analysis_code.canonical import sha,temperature_invariant
from analysis_code.final_acceptance import DEFAULT,verify_outputs

def verify(out=DEFAULT):
    verify_outputs(out)
    rows=read_csv(out/'per_seed.csv');cal=read_csv(out/'calibration_per_seed.csv')
    for r in rows:
        folder=out/f"stage{r['stage']}"/r['run']/r['seed'];z=np.load(folder/'logits.npz')
        logits=torch.from_numpy(z['test']);labels=z['test_labels'];pred=logits.argmax(1).numpy()
        assert np.isfinite(z['test']).all() and z['test'].shape==(10000,10) and z['val'].shape==(5000,10)
        assert np.mean(pred==labels)==float(r['test_accuracy'])
        matrix=np.bincount(labels*10+pred,minlength=100).reshape(10,10)
        assert np.array_equal(matrix,np.loadtxt(folder/'confusion_matrix.csv',delimiter=',',dtype=int))
        classes=read_csv(folder/'per_class_accuracy.csv')
        assert len(classes)==10 and all(float(row['accuracy'])==matrix[i,i]/1000 for i,row in enumerate(classes))
        c=next(c for c in cal if (c['stage'],c['run'],c['seed'])==(r['stage'],r['run'],r['seed']))
        assert temperature_invariant(logits,float(c['temperature']))
        if r['seed']=='20261105':
            maps=np.load(folder/'gradcam_maps.npz')
            for name in ['predicted_4x4','predicted_8x8','true_4x4','true_8x8']:
                hw=int(name.split('_')[-1].split('x')[0]);assert maps[name].shape==(6,hw,hw)
                assert np.isfinite(maps[name]).all()
    robust=out/'robustness'
    complete=read_json(robust/'completion.json')
    for name,digest in complete['files'].items():assert sha(robust/name)==digest
    for name,digest in read_json(robust/'checkpoint_sha256.json').items():assert sha(ROOT/name)==digest
    raw=read_csv(robust/'robustness_per_seed.csv');grouped=read_csv(robust/'robustness_grouped.csv')
    assert len(raw)==450 and len(grouped)==150
    assert len({(r['model'],r['seed'],r['perturbation'],r['level']) for r in raw})==450
    for r in raw:
        stage='2' if r['model'].startswith('stage2_') else '3'
        name=r['model'].removeprefix('stage2_')
        reference=next(x for x in rows if (x['stage'],x['run'],x['seed'])==(stage,name,r['seed']))
        assert abs(float(r['clean_accuracy'])-float(reference['test_accuracy']))<1e-12
        assert abs(float(r['drop_pp'])-100*(float(r['clean_accuracy'])-float(r['perturbed_accuracy'])))<1e-10
    for r in grouped:
        block=[x for x in raw if (x['model'],x['perturbation'],x['level'])==(r['model'],r['perturbation'],r['level'])]
        assert len(block)==3
        for old,new in [('perturbed_accuracy','perturbed_accuracy'),('drop_pp','drop_pp'),('clean_accuracy','clean_accuracy')]:
            values=[float(x[old]) for x in block]
            assert abs(mean(values)-float(r[new+'_mean']))<1e-10
            assert abs(stdev(values)-float(r[new+'_sd']))<1e-10
    for r in read_csv(robust/'robustness_summary.csv'):
        for condition in ['gaussian_noise','gaussian_blur','brightness','contrast','occlusion']:
            block=[x for x in grouped if x['model']==r['model'] and x['perturbation']==condition]
            assert len(block)==5
            assert abs(mean(float(x['perturbed_accuracy_mean']) for x in block)-float(r[condition+'_absolute_accuracy_mean']))<1e-12
    print('Derived semantics verified: 39 logits/confusions/calibrations, 13 native CAM sets, 450 robustness rows')
    return True

if __name__=='__main__':verify()
