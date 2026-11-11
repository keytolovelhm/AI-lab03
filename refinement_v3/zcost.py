"""Maintenance adapter: verified FP32 calibration and pixel_v1 robustness.

The old BF16-clean/FP32-perturbed normalized-space analysis is retired.
Existing historical files are retained for provenance, never silently regenerated.
"""
def stage3_models():
    from .corrected import output_root
    root=output_root()/'experiments'
    files=sorted(root.glob('*/*/best.pt'))
    if len(files)!=21:raise FileNotFoundError(f"Expected 21 corrected checkpoints: {root}")
    return {f"{f.parent.parent.name}/{f.parent.name}":f for f in files}

def run_zero_cost(models='all',out=None):
    from pathlib import Path
    from analysis_code.final_acceptance import DEFAULT,verify_outputs
    from analysis_code import robustness_corrected
    stage3_models();verify_outputs(DEFAULT)
    if models!='all':raise ValueError('Subset legacy analysis retired; use all formal comparisons')
    robustness_corrected.OUT=Path(out) if out else DEFAULT/'robustness'
    if (robustness_corrected.OUT/'completion.json').exists():
        from .common import read_json
        from analysis_code.canonical import sha
        c=read_json(robustness_corrected.OUT/'completion.json')
        for name,d in c['files'].items():
            if sha(robustness_corrected.OUT/name)!=d:raise ValueError(f'Changed robustness file: {name}')
        print('Existing robustness results verified; no overwrite')
        return
    robustness_corrected.main()
