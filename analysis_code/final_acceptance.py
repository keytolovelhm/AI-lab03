"""Read-only audit of all formal checkpoints; derived results never replace training records."""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_code.canonical import load_checkpoint, infer, sha, temperature_invariant, output_directory
from refinement_v3.common import read_json, read_csv, write_json, write_csv
from refinement_v3.calibration import calibration_metrics, fit_temperature

DEFAULT = ROOT / 'refinement_v3/results/final_verified/v1'

def formal_runs():
    for stage, suite, count in [(2, ROOT/'refinement/runs/suite', 18),
                                (3, ROOT/'refinement_v3/runs/suite_corrected', 21)]:
        files = sorted(suite.glob('experiments/*/*/best.pt'))
        if len(files) != count:
            raise ValueError(f'Stage {stage}: expected {count} checkpoints, found {len(files)}')
        manifest = read_json(suite/('evaluation_manifest.json' if stage == 2 else 'test_manifest.json'))
        for checkpoint in files:
            if sha(checkpoint) not in json.dumps(manifest):
                raise ValueError(f'Unsealed checkpoint {checkpoint}')
            yield stage, checkpoint

def verify_formal():
    """No test inference: verify every sealed checkpoint, history, identity, predictions."""
    result = []
    for stage, path in formal_runs():
        folder = path.parent
        model, ck = load_checkpoint(path)
        summary, history = read_json(folder/'training.json'), read_csv(folder/'history.csv')
        best = max(history, key=lambda x: (float(x['val_accuracy']), -float(x['val_loss'])))
        assert len(history) == summary['epochs_completed']
        assert ck['best_epoch'] == summary['best_epoch'] == int(best['epoch'])
        assert ck['config'] == summary['config'] and ck['run'] == summary['run']
        assert ck['seed'] == summary['seed']
        assert float(best['val_accuracy']) == summary['best_val_accuracy']
        assert float(best['val_loss']) == summary['best_val_loss']
        predictions = read_csv(folder/'evaluation/predictions.csv')
        assert len(predictions) == 10000
        assert [int(r['test_index']) for r in predictions] == list(range(10000))
        recorded = read_json(folder/'evaluation/test.json')['test_accuracy']
        assert sum(r['true_class'] == r['predicted_class'] for r in predictions)/10000 == recorded
        result.append((stage, path, summary, history, ck))
    return result

def render(data, logits, history, summary, folder, model=None):
    from src.plots import learning_curves, confusion_plot
    from refinement.visualization import save_errors, save_gradcam
    from types import SimpleNamespace
    history = [{k: (int(v) if k == 'epoch' else float(v)) for k, v in r.items()} for r in history]
    learning_curves(history, summary, folder/'learning_curves.png')
    truth, pred = data.test_labels.numpy(), logits.argmax(1).numpy()
    matrix = np.bincount(truth*10+pred, minlength=100).reshape(10,10)
    assert matrix.sum() == 10000 and np.all(matrix.sum(1) == 1000)
    np.savetxt(folder/'confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
    confusion_plot(matrix, data.classes, summary['run']['name'], folder/'confusion_matrix.png')
    write_csv(folder/'per_class_accuracy.csv', [dict(class_name=c, correct=int(matrix[i,i]), total=1000,
                accuracy=int(matrix[i,i])/1000) for i,c in enumerate(data.classes)])
    if model is not None:
        meta = data.normalization
        raw = ((data.test_images*torch.tensor(meta['train_std'])+torch.tensor(meta['train_mean']))*255)
        raw = raw.round().clamp(0,255).byte().numpy()
        dataset = SimpleNamespace(data=raw, classes=data.classes)
        conf = logits.softmax(1).amax(1).numpy()
        samples = [(i,int(t),int(p),float(c)) for i,(t,p,c) in enumerate(zip(truth,pred,conf))]
        save_errors(dataset, samples, summary['run']['name'], summary['seed'], folder)
        save_gradcam(model, dataset, data.test_images, samples, summary['run']['name'],
                     torch.device('cpu'), folder)

def recompute(out):
    from refinement_v3.data import load_dataset
    from refinement_v3.corrected import validate_split
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    runs = verify_formal()  # All validation-selected checkpoints verified before loading test.
    out = output_directory(out)
    data = load_dataset()
    assert len(data.train_labels) == 45000 and len(data.val_labels) == 5000 and len(data.test_labels) == 10000
    rows, protected, calibration = [], {}, []
    previous = ROOT/'refinement_v3/results/corrected/calibration/calibration_per_seed.csv'
    previous_rows = {(r['run'],int(r['seed'])):r for r in read_csv(previous)} if previous.exists() else {}
    for stage, path, summary, history, ck in runs:
        model, ck = load_checkpoint(path)
        test = infer(model, data.test_images)
        val = infer(model, data.val_images)
        name, seed = ck['run']['name'], ck['seed']
        folder = out/f'stage{stage}'/name/str(seed)
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'logits.npz', test=test.numpy(), val=val.numpy(),
                            test_labels=data.test_labels.numpy(), val_labels=data.val_labels.numpy())
        accuracy = int((test.argmax(1)==data.test_labels).sum())/10000
        row = dict(stage=stage,run=name,seed=seed,test_accuracy=accuracy,
                   val_fp32_accuracy=int((val.argmax(1)==data.val_labels).sum())/5000,
                   historical_bf16_accuracy=read_json(path.parent/'evaluation/test.json')['test_accuracy'],
                   selected_val_bf16=summary['best_val_accuracy'],epochs=len(history),
                   best_epoch=summary['best_epoch'],stop_reason=summary['stop_reason'],
                   seconds=summary['training_seconds'],parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path))
        write_json(folder/'metrics.json',row)
        pred, conf = test.argmax(1), test.softmax(1).amax(1)
        write_csv(folder/'predictions.csv', [dict(test_index=i,true_class=data.classes[int(y)],
                  predicted_class=data.classes[int(p)],confidence=float(c))
                  for i,(y,p,c) in enumerate(zip(data.test_labels,pred,conf))])
        # Every run gets curves/confusion/per-class; one predeclared seed per method gets CAM/errors.
        render(data,test,history,summary,folder,model if seed==20261105 else None)
        fit = fit_temperature(val,data.val_labels)
        t = fit['temperature']
        assert temperature_invariant(test,t)
        raw, cal = calibration_metrics(test,data.test_labels),calibration_metrics(test,data.test_labels,t)
        cr = dict(stage=stage,run=name,seed=seed,temperature=t,accuracy=accuracy,
                  accuracy_after_temperature=accuracy,argmax_preserved=True)
        for k in ['ece','nll','brier','mean_confidence']:
            cr[k+'_raw'],cr[k+'_cal'] = raw[k],cal[k]
        calibration.append(cr)
        if stage==3 and (name,seed) in previous_rows:
            assert abs(float(previous_rows[name,seed]['accuracy'])-accuracy)<1e-7
        for f in [path,path.parent/'training.json',path.parent/'history.csv',
                  path.parent/'evaluation/test.json',path.parent/'evaluation/predictions.csv']:
            protected[str(f.relative_to(ROOT))] = sha(f)
        rows.append(row)
        print(f'FP32 stage{stage} {name}/{seed}: {accuracy:.4%}',flush=True)
    write_csv(out/'per_seed.csv',rows)
    write_csv(out/'calibration_per_seed.csv',calibration)
    groups=[]
    for stage,name in sorted({(r['stage'],r['run']) for r in rows}):
        block=[r for r in rows if r['stage']==stage and r['run']==name]
        assert len(block)==3 and len({r['seed'] for r in block})==3
        g=dict(stage=stage,run=name,n_seeds=3)
        for key in ['test_accuracy','val_fp32_accuracy','seconds']:
            g[key+'_mean']=mean(r[key] for r in block);g[key+'_sd']=stdev(r[key] for r in block)
        groups.append(g)
    write_csv(out/'grouped.csv',groups)
    pairs=[]
    for row in rows:
        if row['stage']!=3:continue
        reference='baseline_plain' if row['run']=='convstem_plain' else 'baseline_both'
        baseline=next(r for r in rows if r['stage']==2 and r['run']==reference and r['seed']==row['seed'])
        pairs.append(dict(run=row['run'],seed=row['seed'],reference=reference,
                          difference_pp=100*(row['test_accuracy']-baseline['test_accuracy']),
                          same_split=True,same_seed=True,stopping_algorithm_same=True,
                          suite_extension_decision_independent=True,identical_training_trajectory=False))
    write_csv(out/'paired_differences.csv',pairs)
    write_json(out/'formal_sha256.json',protected)
    write_json(out/'protocol.json',dict(device='cpu',precision='fp32',autocast=False,batch_size=512,
        model_mode='eval',model_layout='channels_last',loader='checkpoint config/run + strict state_dict keys',
        split_sha256=validate_split(),temperature_fit='validation only',ece_bins=15,
        aggregation='sample SD ddof=1 across 3 training seeds, never pooled as independent test samples',
        test_used_for='reporting; no training or selection',n_formal_runs=len(rows)))
    write_json(out/'completion.json',dict(formal_runs=len(rows),files={str(f.relative_to(out)):sha(f)
               for f in out.rglob('*') if f.is_file()}))
    return rows

def verify_outputs(out):
    c=read_json(out/'completion.json')
    for f,d in c['files'].items():
        assert sha(out/f)==d, f'Changed derived result: {f}'
    for f,d in read_json(out/'formal_sha256.json').items():
        assert sha(ROOT/f)==d, f'Changed formal result: {f}'
    return c['formal_runs']

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','recompute','summary'])
    p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args()
    if a.action=='recompute':recompute(a.out)
    elif a.action=='verify':
        torch.set_num_threads(8);print('formal:',len(verify_formal()),'derived:',verify_outputs(a.out))
    else:
        verify_outputs(a.out)
        for r in read_csv(a.out/'grouped.csv'):
            print(f"stage{r['stage']} {r['run']}: {100*float(r['test_accuracy_mean']):.3f} +/- {100*float(r['test_accuracy_sd']):.3f}%")

if __name__=='__main__':main()
