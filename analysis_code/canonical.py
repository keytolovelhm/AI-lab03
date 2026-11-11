"""Strict, checkpoint-defined CPU FP32 evaluation; no positional tensor loading."""
from pathlib import Path
import hashlib
import torch

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 512

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_checkpoint(path):
    from refinement_v3.models import build_stage_model
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if path.parent.name != str(ck['seed']) or path.parent.parent.name != ck['run']['name']:
        raise ValueError(f"Checkpoint identity disagrees with path: {path}")
    model = build_stage_model(ck['config'], ck['run']).to(dtype=torch.float32,
                                                       memory_format=torch.channels_last)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, ck

@torch.inference_mode()
def infer(model, images):
    if torch.is_autocast_enabled('cpu'):
        raise ValueError('Canonical FP32 inference cannot run inside autocast')
    model.eval()
    return torch.cat([model(x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
                           .to(dtype=torch.float32)).float().cpu()
                      for x in images.split(BATCH_SIZE)])

def temperature_invariant(logits, temperature):
    import math
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return bool(torch.equal(logits.argmax(1), (logits / temperature).argmax(1)))

def output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Results are preserved: {path}. Choose a new --out directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path
