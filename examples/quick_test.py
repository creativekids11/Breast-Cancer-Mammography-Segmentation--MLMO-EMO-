"""Quick test of MLMO-EMO model"""
import torch
from mlmo_emo_segmentation import create_mlmo_emo_model, MLMOEMOLoss

print("✓ Imports successful")

print("\nCreating model...")
model = create_mlmo_emo_model(num_particles=2, emo_iterations=2)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

print("\nTesting forward pass...")
dummy_input = torch.randn(1, 3, 256, 256)  # Smaller size for faster test
with torch.no_grad():
    outputs = model(dummy_input)
print(f"✓ Forward pass successful")
print(f"  - Segmentation shape: {outputs['segmentation'].shape}")
print(f"  - Multi-objective outputs: {list(outputs['multi_objective'].keys())}")

print("\nTesting loss calculation...")
loss_fn = MLMOEMOLoss()
dummy_target = torch.randint(0, 2, (1, 1, 256, 256)).float()
losses = loss_fn(outputs, dummy_target)
print(f"✓ Loss calculation successful")
print(f"  - Total loss: {losses['total'].item():.4f}")

print("\n✅ All tests passed! MLMO-EMO model is working correctly.")
