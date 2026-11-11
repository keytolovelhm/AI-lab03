"""Read-only verification, separate from training-time source provenance."""
def verify_all():
    from analysis_code.final_acceptance import DEFAULT,verify_formal,verify_outputs
    count=len(verify_formal());derived=verify_outputs(DEFAULT)
    print(f"formal={count}, derived={derived}")
    return count==derived==39
