"""Compatibility name retained; positional shape-only loading is no longer allowed."""
import torch
def load_by_shape(model: torch.nn.Module, state: dict) -> torch.nn.Module:
    """Require exact semantic keys and tensor shapes. Reordered dictionaries are safe."""
    model.load_state_dict(state, strict=True)
    return model
