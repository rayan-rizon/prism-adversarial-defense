"""
TAMM: Activation Extractor
Extracts intermediate activations from checkpoint layers of any PyTorch model
using forward hooks. Architecture-agnostic by design.
"""
import torch
from typing import Dict, List, Optional


class ActivationExtractor:
    """Extracts activations from named layers of any PyTorch model via hooks."""

    def __init__(self, model: torch.nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all target layers."""
        module_dict = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in module_dict:
                available = list(module_dict.keys())[:20]
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available (first 20): {available}"
                )
            layer = module_dict[name]
            hook = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # Detach to avoid gradient graph retention; keep on same device
            self.activations[name] = output.detach()
        return hook_fn

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and return dict of {layer_name: activation_tensor}."""
        self.activations = {}
        with torch.no_grad():
            self.model(x)
        # Verify all layers were captured
        missing = set(self.layer_names) - set(self.activations.keys())
        if missing:
            raise RuntimeError(
                f"Hooks failed to capture layers: {missing}. "
                f"Ensure the model forward pass traverses these layers."
            )
        return dict(self.activations)

    def cleanup(self):
        """Remove all hooks. Call when done to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.cleanup()
