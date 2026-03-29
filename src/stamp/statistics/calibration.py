"""Calibration metrics for evaluating model confidence.

This module implements Expected Calibration Error (ECE) and reliability diagrams
for assessing and visualizing the calibration of classification models.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks.
    In International Conference on Machine Learning (pp. 1321-1330). PMLR.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt

__author__ = "Philipp Kuhn"
__copyright__ = "Copyright (C) 2026 Philipp Kuhn"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


def compute_ece(
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual accuracy.
    Lower ECE indicates better calibration.

    Args:
        probs: Predicted probabilities for the positive class (shape: [n_samples])
               or for all classes (shape: [n_samples, n_classes]).
        labels: Ground truth labels (shape: [n_samples]).
        n_bins: Number of bins for computing ECE (default: 10).

    Returns:
        Tuple of:
            - ECE value (float between 0 and 1)
            - Bin accuracies (shape: [n_bins])
            - Bin confidences (shape: [n_bins])
            - Bin counts (shape: [n_bins])
    """
    # Handle multi-class probabilities
    if probs.ndim == 2:
        # Get max probability and corresponding prediction
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        # Binary classification
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    # Ensure labels are integers
    labels = labels.astype(int)

    # Bin boundaries (equal width)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    # Initialize arrays
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Compute accuracy and confidence for each bin
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(predictions[in_bin] == labels[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])

    # Compute ECE
    n_samples = len(labels)
    ece = np.sum((bin_counts / n_samples) * np.abs(bin_accuracies - bin_confidences))

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    output_path: Path,
    title: str = "Reliability Diagram",
    class_name: str | None = None,
) -> float:
    """Plot a reliability diagram and save to file.

    A reliability diagram shows the relationship between predicted confidence
    and actual accuracy. Perfect calibration is represented by the diagonal line.

    Args:
        probs: Predicted probabilities for the positive class (shape: [n_samples])
               or for all classes (shape: [n_samples, n_classes]).
        labels: Ground truth labels (shape: [n_samples]).
        n_bins: Number of bins for the diagram (default: 10).
        output_path: Path to save the SVG file.
        title: Title for the plot.
        class_name: Optional class name for the title.

    Returns:
        ECE value for the predictions.
    """
    ece, bin_accuracies, bin_confidences, bin_counts = compute_ece(
        probs=probs, labels=labels, n_bins=n_bins
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    # Plot reliability curve
    bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins

    # Only plot bins with samples
    valid_bins = bin_counts > 0
    ax.plot(
        bin_confidences[valid_bins],
        bin_accuracies[valid_bins],
        "o-",
        color="steelblue",
        markersize=8,
        linewidth=2,
        label=f"Model (ECE={ece:.4f})",
    )

    # Add histogram of predictions
    ax2 = ax.twinx()
    ax2.bar(
        bin_centers,
        bin_counts,
        width=1.0 / n_bins,
        alpha=0.3,
        color="gray",
        label="Sample count",
    )
    ax2.set_ylabel("Sample count", fontsize=12)

    # Labels and title
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    plot_title = title
    if class_name:
        plot_title = f"{title} - {class_name}"
    ax.set_title(plot_title, fontsize=14)

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    _logger.info(f"Reliability diagram saved to: {output_path}")
    _logger.info(f"ECE: {ece:.4f}")

    return ece


def compute_calibration_metrics(
    *,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute multiple calibration metrics.

    Args:
        probs: Predicted probabilities for the positive class (shape: [n_samples])
               or for all classes (shape: [n_samples, n_classes]).
        labels: Ground truth labels (shape: [n_samples]).
        n_bins: Number of bins for computing metrics.

    Returns:
        Dictionary with calibration metrics:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - brier_score: Brier score (mean squared error of probabilities)
    """
    ece, bin_accuracies, bin_confidences, bin_counts = compute_ece(
        probs=probs, labels=labels, n_bins=n_bins
    )

    # Maximum Calibration Error (MCE)
    valid_bins = bin_counts > 0
    mce = float(np.max(np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])))

    # Brier score
    if probs.ndim == 2:
        # Multi-class: use one-hot encoding
        n_classes = probs.shape[1]
        one_hot_labels = np.zeros((len(labels), n_classes))
        one_hot_labels[np.arange(len(labels)), labels] = 1
        brier_score = float(np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1)))
    else:
        # Binary
        brier_score = float(np.mean((probs - labels) ** 2))

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier_score,
    }