"""Temperature scaling calibration for model confidence calibration.

This module implements temperature scaling, a post-hoc calibration method
that learns a single scalar parameter to scale logits before softmax,
improving the calibration of model probabilities without affecting accuracy.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks.
    In International Conference on Machine Learning (pp. 1321-1330). PMLR.
"""

import logging
from pathlib import Path
from typing import TypeAlias

import lightning
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

from stamp.modeling.config import CalibrationConfig
from stamp.types import Bags, BagSizes, CoordinatesBatch, EncodedTargets

__author__ = "Philipp Kuhn"
__copyright__ = "Copyright (C) 2026 Philipp Kuhn"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

Logits: TypeAlias = Tensor
Temperature: TypeAlias = float


class TemperatureScaler(nn.Module):
    """Temperature scaling module for calibrating model predictions.

    This module wraps a trained model and applies a learned temperature
    parameter to scale logits before softmax. The temperature is optimized
    on a validation set using negative log-likelihood loss.

    Attributes:
        model: The trained model to calibrate.
        temperature: Learnable temperature parameter (initialized to 1.0).
    """

    def __init__(self, model: lightning.LightningModule) -> None:
        """Initialize the temperature scaler.

        Args:
            model: A trained PyTorch Lightning model.
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass with temperature scaling applied to logits.

        Returns:
            Temperature-scaled logits (logits / temperature).
        """
        # Get logits from the model
        logits = self.model(*args, **kwargs)
        # Apply temperature scaling
        return logits / self.temperature

    def predict_with_temperature(self, *args, **kwargs) -> Tensor:
        """Get calibrated probabilities after temperature scaling.

        Returns:
            Calibrated probabilities (softmax of scaled logits).
        """
        scaled_logits = self.forward(*args, **kwargs)
        return F.softmax(scaled_logits, dim=-1)

    def calibrate(
        self,
        valid_dl: DataLoader,
        max_iterations: int = 50,
        learning_rate: float = 0.01,
    ) -> Temperature:
        """Optimize the temperature parameter on validation data.

        Uses NLL loss to find the optimal temperature that minimizes
        the negative log-likelihood of the validation set predictions.

        Args:
            valid_dl: DataLoader for validation data.
            max_iterations: Maximum number of optimization iterations.
            learning_rate: Learning rate for temperature optimization.

        Returns:
            The optimized temperature value.
        """
        self.model.eval()
        self.train()  # Only temperature is trainable

        # Collect all logits and targets first
        all_logits: list[Tensor] = []
        all_targets: list[Tensor] = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in valid_dl:
                logits, targets = self._get_logits_and_targets(batch, device)
                all_logits.append(logits)
                all_targets.append(targets)

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Optimize temperature
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=learning_rate, max_iter=max_iterations
        )

        def eval_loss() -> Tensor:
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Clamp temperature to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)

        final_temperature = self.temperature.item()
        _logger.info(f"Temperature scaling completed. Optimal temperature: {final_temperature:.4f}")

        return final_temperature

    def _get_logits_and_targets(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets] | list[Tensor],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Extract logits and targets from a batch.

        Handles both tile-level (bags) and patient/slide-level (features) batches.
        Also handles sample weights that may be included in the batch.

        Args:
            batch: A batch from the dataloader.
            device: Device to run computation on.

        Returns:
            Tuple of (logits, targets) tensors.
        """
        self.model.eval()

        with torch.no_grad():
            # Check batch structure based on number of elements
            # 5 elements: tile-level with sample weights (bags, coords, bag_sizes, targets, weights)
            # 4 elements: tile-level without weights (bags, coords, bag_sizes, targets)
            # 3 elements: patient/slide-level with sample weights (feats, targets, weights)
            # 2 elements: patient/slide-level without weights (feats, targets)
            if len(batch) == 5:
                # Tile-level with sample weights: (bags, coords, bag_sizes, targets, weights)
                bags, coords, bag_sizes, targets, weights = batch
                bags = bags.to(device)
                coords = coords.to(device)
                # Get logits from model
                logits = self.model(bags, coords=coords, mask=None)
            elif len(batch) == 4:
                # Tile-level without weights: (bags, coords, bag_sizes, targets)
                bags, coords, bag_sizes, targets = batch
                bags = bags.to(device)
                coords = coords.to(device)
                # Get logits from model
                logits = self.model(bags, coords=coords, mask=None)
            elif len(batch) == 3:
                # Patient/slide-level with sample weights: (feats, targets, weights)
                feats, targets, weights = batch
                feats = feats.to(device)
                logits = self.model(feats)
            else:
                # Patient/slide-level without weights: (feats, targets)
                feats, targets = batch
                feats = feats.to(device)
                logits = self.model(feats)

            # Convert one-hot targets to class indices if needed
            if targets.dim() > 1 and targets.size(1) > 1:
                targets = targets.argmax(dim=1)
            targets = targets.to(device)

        return logits, targets


def calibrate_model_(
    *,
    model: lightning.LightningModule,
    valid_dl: DataLoader,
    output_dir: Path,
    config: CalibrationConfig,
) -> Path:
    """Calibrate a trained model using temperature scaling.

    This function applies temperature scaling to a trained model and saves
    the calibrated model checkpoint alongside the original.

    Args:
        model: The trained Lightning model to calibrate.
        valid_dl: DataLoader for validation data (used for calibration).
        output_dir: Directory to save the calibrated checkpoint.
        config: Calibration configuration.

    Returns:
        Path to the calibrated checkpoint file.
    """
    if not config.enabled:
        _logger.info("Temperature scaling calibration is disabled, skipping.")
        return output_dir / "model.ckpt"

    _logger.info("Starting temperature scaling calibration...")

    # Create temperature scaler
    scaler = TemperatureScaler(model)

    # Calibrate on validation set
    optimal_temperature = scaler.calibrate(
        valid_dl=valid_dl,
        max_iterations=config.max_iterations,
        learning_rate=config.learning_rate,
    )

    # Store temperature in model hyperparameters
    model.hparams["temperature"] = optimal_temperature
    model.hparams["calibrated"] = True

    # Save calibrated checkpoint
    calibrated_path = output_dir / "model_calibrated.ckpt"
    model.save_checkpoint(calibrated_path)
    _logger.info(f"Calibrated model saved to: {calibrated_path}")

    return calibrated_path


def apply_temperature_scaling(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Model logits tensor.
        temperature: Temperature parameter.

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def get_calibrated_probabilities(logits: Tensor, temperature: float) -> Tensor:
    """Get calibrated probabilities from logits using temperature scaling.

    Args:
        logits: Model logits tensor.
        temperature: Temperature parameter.

    Returns:
        Calibrated probabilities (softmax of scaled logits).
    """
    scaled_logits = apply_temperature_scaling(logits, temperature)
    return F.softmax(scaled_logits, dim=-1)