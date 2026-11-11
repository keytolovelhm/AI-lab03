"""Summarise verified FP32 formal results; the deleted legacy suite is never consulted."""
def build_summary():
    from analysis_code.final_acceptance import DEFAULT,verify_outputs
    from .common import read_csv
    verify_outputs(DEFAULT)
    rows=read_csv(DEFAULT/'grouped.csv')
    for r in rows:
        print(f"stage{r['stage']} {r['run']}: {100*float(r['test_accuracy_mean']):.3f} +/- {100*float(r['test_accuracy_sd']):.3f}%")
    return rows
