"""
Multi-Level Multi-Objective Electromagnetism-like Optimization (MLMO-EMO)
for Breast Cancer Mammography Segmentation

This implementation uses electromagnetism-like optimization principles combined with
deep learning for direct whole-image segmentation.

Key Components:
1. Multi-level feature extraction (different scales)
2. Electromagnetism-like optimization (attraction-repulsion forces)
3. Multi-objective optimization (accuracy, boundary quality, region homogeneity)
4. Direct end-to-end segmentation without cascading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import segmentation_models_pytorch as smp


class ElectromagneticParticle(nn.Module):
    """
    Lightweight electromagnetic particle for efficient segmentation optimization.
    Reduced computational complexity while maintaining core functionality.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        # Simplified position encoder - single conv layer
        self.position_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight charge calculation
        self.charge_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Simplified force calculation - lightweight
        self.force_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features [B, C, H, W]
        Returns:
            position: Encoded position [B, hidden_dim, H, W]
            charge: Particle charge [B, 1, H, W]
        """
        position = self.position_encoder(features)
        charge = self.charge_net(position)
        return position, charge
    
    def compute_force(self, pos1: torch.Tensor, pos2: torch.Tensor, 
                     charge1: torch.Tensor, charge2: torch.Tensor) -> torch.Tensor:
        """
        Compute electromagnetic force between particles.
        Attraction for similar regions, repulsion for boundaries.
        """
        # Concatenate for force computation
        combined = torch.cat([pos1, pos2], dim=1)
        force = self.force_net(combined)
        
        # Modulate by charges (average charge strength)
        charge_modulation = (charge1 + charge2) / 2.0
        force = force * charge_modulation
        
        return force


class MultiLevelFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales/levels for comprehensive image understanding.
    """
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"):
        super().__init__()
        
        # Use pretrained encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder output channels for each level
        self.encoder_channels = self.encoder.out_channels
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-level features.
        Returns list of features from different encoder stages.
        """
        features = self.encoder(x)
        return features  # Returns features at multiple scales


class MultiObjectiveOptimizationModule(nn.Module):
    """
    Lightweight multi-objective optimization for efficiency.
    Simplified version with shared backbone.
    """
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        
        # Shared backbone to reduce computation
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        shared_channels = in_channels // 4
        
        # Lightweight objective heads
        self.accuracy_head = nn.Conv2d(shared_channels, num_classes, 1)
        self.boundary_head = nn.Conv2d(shared_channels, num_classes, 1)
        self.homogeneity_head = nn.Conv2d(shared_channels, num_classes, 1)
        
        # Learnable weights for combining objectives
        self.objective_weights = nn.Parameter(torch.ones(3) / 3.0)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all objectives with shared backbone for efficiency.
        """
        shared_features = self.shared_backbone(features)
        
        accuracy_out = self.accuracy_head(shared_features)
        boundary_out = self.boundary_head(shared_features)
        homogeneity_out = self.homogeneity_head(shared_features)
        
        # Normalize weights
        weights = F.softmax(self.objective_weights, dim=0)
        
        # Combined output
        combined = (weights[0] * accuracy_out + 
                   weights[1] * boundary_out + 
                   weights[2] * homogeneity_out)
        
        return {
            'combined': combined,
            'accuracy': accuracy_out,
            'boundary': boundary_out,
            'homogeneity': homogeneity_out,
            'weights': weights
        }


class MLMOEMOSegmentationModel(nn.Module):
    """
    Complete Multi-Level Multi-Objective Electromagnetism-like Optimization Model
    for whole-image breast cancer segmentation.
    """
    def __init__(self,
                 encoder_name: str = "resnet18",
                 encoder_weights: str = "imagenet",
                 num_classes: int = 1,
                 num_particles: int = 2,
                 hidden_dim: int = 64,
                 emo_iterations: int = 2):
        super().__init__()

        self.num_particles = num_particles
        self.emo_iterations = emo_iterations
        self.hidden_dim = hidden_dim

        # Multi-level feature extraction
        self.feature_extractor = MultiLevelFeatureExtractor(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )

        encoder_channels = self.feature_extractor.encoder_channels

        # Decoder channels (kept moderate to allow smaller models)
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = nn.ModuleList()

        decoder_in_channels = [
            512,
            256 + 256,
            128 + 128,
            64 + 64,
            32 + 64
        ]

        for in_ch, out_ch in zip(decoder_in_channels, self.decoder_channels):
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.decoder.append(block)

        decoder_out_channels = 16

        # Electromagnetic particles for optimization
        self.particles = nn.ModuleList([
            ElectromagneticParticle(decoder_out_channels, hidden_dim)
            for _ in range(num_particles)
        ])

        # Feature fusion after EMO
        self.emo_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * num_particles, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Multi-objective optimization module
        self.multi_objective = MultiObjectiveOptimizationModule(
            in_channels=hidden_dim,
            num_classes=num_classes
        )

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        
    def apply_electromagnetic_optimization(
        self, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply electromagnetism-like optimization through particle interactions.
        """
        B, C, H, W = features.shape
        
        # Initialize particles
        particle_positions = []
        particle_charges = []
        
        for particle in self.particles:
            pos, charge = particle(features)
            particle_positions.append(pos)
            particle_charges.append(charge)
        
        # Iterative EMO refinement
        for iteration in range(self.emo_iterations):
            refined_positions = []
            
            for i, (pos_i, charge_i, particle) in enumerate(
                zip(particle_positions, particle_charges, self.particles)
            ):
                # Compute forces from other particles
                total_force = torch.zeros_like(pos_i)
                
                for j, (pos_j, charge_j) in enumerate(
                    zip(particle_positions, particle_charges)
                ):
                    if i != j:
                        force = particle.compute_force(pos_i, pos_j, charge_i, charge_j)
                        total_force = total_force + force
                
                # Update position based on forces
                refined_pos = pos_i + 0.1 * total_force  # Learning rate for force
                refined_positions.append(refined_pos)
            
            particle_positions = refined_positions
        
        # Fuse all particle information
        fused = torch.cat(particle_positions, dim=1)
        return self.emo_fusion(fused)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MLMO-EMO segmentation.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Dictionary with segmentation outputs and intermediate results
        """
        input_size = x.shape[-2:]
        
        # Multi-level feature extraction
        encoder_features = self.feature_extractor(x)
        # encoder_features: [0: 3x512x512, 1: 64x256x256, 2: 64x128x128, 
        #                    3: 128x64x64, 4: 256x32x32, 5: 512x16x16]
        
        # Start decoding from the deepest features
        decoder_output = encoder_features[5]  # 512x16x16
        
        # Decoder block 0: 512 -> 256, then upsample to 32x32
        decoder_output = self.decoder[0](decoder_output)
        decoder_output = F.interpolate(decoder_output, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder block 1: concat with encoder[4] (256x32x32), then 512 -> 128, upsample to 64x64
        decoder_output = torch.cat([decoder_output, encoder_features[4]], dim=1)
        decoder_output = self.decoder[1](decoder_output)
        decoder_output = F.interpolate(decoder_output, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder block 2: concat with encoder[3] (128x64x64), then 256 -> 64, upsample to 128x128
        decoder_output = torch.cat([decoder_output, encoder_features[3]], dim=1)
        decoder_output = self.decoder[2](decoder_output)
        decoder_output = F.interpolate(decoder_output, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder block 3: concat with encoder[2] (64x128x128), then 128 -> 32, upsample to 256x256
        decoder_output = torch.cat([decoder_output, encoder_features[2]], dim=1)
        decoder_output = self.decoder[3](decoder_output)
        decoder_output = F.interpolate(decoder_output, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Decoder block 4: concat with encoder[1] (64x256x256), then 96 -> 16, upsample to 512x512
        decoder_output = torch.cat([decoder_output, encoder_features[1]], dim=1)
        decoder_output = self.decoder[4](decoder_output)
        decoder_output = F.interpolate(decoder_output, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Apply electromagnetic optimization
        emo_features = self.apply_electromagnetic_optimization(decoder_output)
        
        # Multi-objective optimization
        multi_obj_outputs = self.multi_objective(emo_features)
        
        # Final segmentation
        final_seg = self.segmentation_head(emo_features)
        
        # Upsample all outputs to input size
        for key in multi_obj_outputs:
            if key != 'weights':
                multi_obj_outputs[key] = F.interpolate(
                    multi_obj_outputs[key],
                    size=input_size,
                    mode='bilinear',
                    align_corners=False
                )
        
        final_seg = F.interpolate(
            final_seg,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        
        return {
            'segmentation': final_seg,
            'multi_objective': multi_obj_outputs,
            'emo_features': emo_features
        }


class MLMOEMOLoss(nn.Module):
    """
    Custom loss function for MLMO-EMO model with L1 regularization to prevent overfitting.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        boundary_weight: float = 0.3,
        homogeneity_weight: float = 0.2,
        l1_weight: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.homogeneity_weight = homogeneity_weight
        self.l1_weight = l1_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for segmentation."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage smooth boundaries."""
        pred = torch.sigmoid(pred)
        
        # Compute gradients for boundary detection
        pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # L2 loss on boundaries
        loss_x = F.mse_loss(pred_dx, target_dx)
        loss_y = F.mse_loss(pred_dy, target_dy)
        
        return loss_x + loss_y
    
    def homogeneity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage homogeneous regions."""
        pred = torch.sigmoid(pred)
        
        # Compute variance within predicted regions
        pred_variance = torch.var(pred * target, dim=(2, 3)).mean()
        
        return pred_variance
    
    def l1_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute L1 regularization loss to prevent overfitting."""
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        model: nn.Module = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining all objectives.
        """
        seg_output = outputs['segmentation']
        multi_obj = outputs['multi_objective']
        
        # Main segmentation losses
        dice = self.dice_loss(seg_output, target)
        bce = self.bce_loss(seg_output, target)
        
        # Multi-objective losses
        boundary = self.boundary_loss(multi_obj['boundary'], target)
        homogeneity = self.homogeneity_loss(multi_obj['homogeneity'], target)
        
        # L1 regularization loss (if model provided)
        l1_loss = 0
        if model is not None and self.l1_weight > 0:
            l1_loss = self.l1_regularization(model)
        
        # Combined loss
        total_loss = (
            self.dice_weight * dice +
            self.bce_weight * bce +
            self.boundary_weight * boundary +
            self.homogeneity_weight * homogeneity +
            self.l1_weight * l1_loss
        )
        
        return {
            'total': total_loss,
            'dice': dice,
            'bce': bce,
            'boundary': boundary,
            'homogeneity': homogeneity,
            'l1': l1_loss
        }


def create_mlmo_emo_model(
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes: int = 1,
    num_particles: int = 2,
    emo_iterations: int = 2,
    hidden_dim: int = 64
) -> MLMOEMOSegmentationModel:
    """
    Factory function to create MLMO-EMO model.
    """
    model = MLMOEMOSegmentationModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_classes=num_classes,
        num_particles=num_particles,
        emo_iterations=emo_iterations,
        hidden_dim=hidden_dim
    )
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Creating MLMO-EMO Segmentation Model...")
    model = create_mlmo_emo_model()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Segmentation output shape: {outputs['segmentation'].shape}")
    print(f"Multi-objective outputs: {list(outputs['multi_objective'].keys())}")
    print(f"Objective weights: {outputs['multi_objective']['weights']}")
    
    # Test loss computation
    loss_fn = MLMOEMOLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 512, 512)).float()
    losses = loss_fn(outputs, dummy_target)
    
    print(f"\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
