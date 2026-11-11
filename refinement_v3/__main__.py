"""Current CLI: historical outputs are immutable, maintenance training is opt-in."""
import argparse
import os
import sys
from .common import SEEDS,V3

def main():
    parser=argparse.ArgumentParser(prog="refinement_v3")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["test","protect","summary","verify","qualitative"]:sub.add_parser(name)
    for name in ["evaluate","zcost"]:
        p=sub.add_parser(name);p.add_argument("--out",type=str)
    p=sub.add_parser("train")
    from .config import P0,P1
    p.add_argument("--run",choices=[r['name'] for r in P0+P1]);p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--group",choices=['all','p0','p1'],default='all')
    p.add_argument("--output",choices=['suite_maintenance'],default='suite_maintenance')
    # Retired actions fail explicitly before touching data or outputs.
    for name in ["snapshot","probe","check"]:sub.add_parser(name)
    a=parser.parse_args()
    if a.command in {'snapshot','probe','check'}:
        parser.error("Retired legacy action. Use test / protect / verify; no runs/suite fallback.")
    if a.command=='test':
        import unittest
        tests=unittest.TestLoader().discover(str(V3/'tests'),top_level_dir=str(V3.parent))
        if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():sys.exit(1)
    elif a.command=='train':
        os.environ['LAB3_SUITE']=a.output
        from .corrected import train
        train(a.run,a.seed,a.group)
    elif a.command=='protect':
        from .guard import check_gate
        print(check_gate())
    elif a.command in {'summary','verify','qualitative'}:
        from analysis_code.final_acceptance import DEFAULT,verify_formal,verify_outputs
        if a.command=='verify':
            print('formal:',len(verify_formal()),'derived:',verify_outputs(DEFAULT))
            from analysis_code.verify_derived import verify
            verify()
        elif a.command=='qualitative':
            verify_outputs(DEFAULT)
            for path in DEFAULT.glob('stage*/*/20261105/gradcam_4x4_vs_8x8.png'):print(path)
        else:
            from .summary import build_summary
            build_summary()
    elif a.command=='evaluate':
        from analysis_code.final_acceptance import DEFAULT,recompute,verify_outputs
        from pathlib import Path
        output=Path(a.out) if a.out else DEFAULT
        if (output/'completion.json').exists():print('Already complete and verified:',verify_outputs(output))
        else:recompute(output)
    else:
        from .zcost import run_zero_cost
        run_zero_cost(out=a.out)

if __name__=='__main__':main()
