"""Reproduce original BF16 test predictions with their own saved protocol, without overwriting."""
import argparse,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis_code.final_acceptance import formal_runs
from analysis_code.canonical import sha
from refinement_v3.common import read_csv,read_json,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    from refinement_v3.data import load_dataset
    from refinement_v3.models import build_stage_model
    from refinement_v3.train import logits_for
    from src.models import build_model
    from src.evaluate import predict
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    files=list(formal_runs());data=load_dataset();rows=[]
    for stage,path in files:
        ck=torch.load(path,map_location='cpu',weights_only=True);device=torch.device('cpu')
        model=(build_model(ck['config'],ck['run'],device) if stage==2 else
               build_stage_model(ck['config'],ck['run']).to(memory_format=torch.channels_last))
        model.load_state_dict(ck['model'],strict=True);model.eval()
        if stage==2:
            cfg={**ck['config'],'batch_size':ck['config']['evaluation_batch_size']}
            logits,pred,_=predict(model,data.test_images,data.test_labels,cfg,device)
        else:
            logits=logits_for(model,data.test_images,ck['config'],device);pred=logits.argmax(1)
        old=read_csv(path.parent/'evaluation/predictions.csv')
        actual=[data.classes[int(p)] for p in pred]
        disagreements=sum(a!=b['predicted_class'] for a,b in zip(actual,old))
        accuracy=int((pred==data.test_labels).sum())/10000
        recorded=read_json(path.parent/'evaluation/test.json')['test_accuracy']
        rows.append(dict(stage=stage,run=ck['run']['name'],seed=ck['seed'],checkpoint_sha256=sha(path),
                         reproduced=accuracy,recorded=recorded,prediction_disagreements=disagreements))
        print(stage,ck['run']['name'],ck['seed'],accuracy,disagreements,flush=True)
        assert accuracy==recorded and disagreements==0
    write_json(a.out,dict(protocol='saved BF16, original stage2/stage3 inference functions',rows=rows,
                           all_predictions_exact=True))

if __name__=='__main__':main()
