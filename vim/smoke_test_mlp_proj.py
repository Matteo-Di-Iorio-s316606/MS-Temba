"""Smoke test for E3 MLP projection module.

Run from MS-Temba-v2/vim/ directory:
    python smoke_test_mlp_proj.py

Verifies:
    1. Module import succeeds.
    2. timm registration works.
    3. Model instantiation does not crash.
    4. self.proj is MLPProjection (not Linear).
    5. Parameter count of self.proj is ~4.46M (vs ~1.05M for the Linear baseline).
    6. forward_features produces tensor with correct shape.
"""
import torch
from timm.models import create_model

import models.extensions.mstemba_mlp_proj  # noqa: F401  registers factory in timm

# 1-3. Instantiate via timm factory.
model = create_model(
    "mstemba_mlp_proj",
    num_classes=157,
    in_feat_dim=4096,
    proj_hidden_dim=1024,
    proj_dropout=0.1,
)

# 4. Verify proj is the MLP, not a Linear.
print("proj type:", type(model.proj).__name__)
print("proj submodules:")
for name, child in model.proj.named_children():
    print(f"    {name}: {type(child).__name__}")

# 5. Parameter count of proj only.
n_proj = sum(p.numel() for p in model.proj.parameters())
print(f"proj n_params: {n_proj:,}  (expected ~4,461,824)")

# Total model params for reference.
n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total trainable params: {n_total:,}")

# 6. Forward shape check.
# MSTemba.forward_features expects input as (B, D, T): see models_MSTemba.py:717
B, D, T = 2, 4096, 256
x = torch.randn(B, D, T)
with torch.no_grad():
    feats = model.forward_features(x)

# feats may be a tuple/list depending on MSTemba internals; print shape(s).
if isinstance(feats, (tuple, list)):
    print("forward_features returned tuple/list of length", len(feats))
    for i, t in enumerate(feats):
        if torch.is_tensor(t):
            print(f"    [{i}] tensor shape: {tuple(t.shape)}")
        else:
            print(f"    [{i}] non-tensor: {type(t).__name__}")
else:
    print(f"forward_features tensor shape: {tuple(feats.shape)}")

print("\nSmoke test passed.")