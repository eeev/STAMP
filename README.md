# Implementation Changes: Model Calibration & MILAN-Weighted Training

> **Context:** These changes extend the [STAMP](https://github.com/KatherLab/STAMP) computational pathology pipeline with two experimental features for a master's thesis project. They target the **classification** task path (binary and multi-class) and are implemented on top of the existing training, cross-validation, deployment, and statistics modules.

---

## Table of Contents

1. [Feature 1: Post-Hoc Model Calibration via Temperature Scaling](#feature-1-post-hoc-model-calibration-via-temperature-scaling)
2. [Feature 2: MILAN-Weighted Training Loss](#feature-2-milan-weighted-training-loss)
3. [Files Modified](#files-modified)
4. [Files Added](#files-added)
5. [Architectural Overview](#architectural-overview)
6. [Detailed Change Log](#detailed-change-log)
7. [Scope and Limitations](#scope-and-limitations)

---

## Feature 1: Post-Hoc Model Calibration via Temperature Scaling

### Motivation

Modern deep neural networks, while achieving high discriminative performance, are frequently poorly calibrated — their predicted class probabilities do not accurately reflect the true likelihood of correctness. In clinical decision support, well-calibrated confidence scores are essential for trustworthy predictions. Temperature scaling, as evaluated by Guo et al. (2017), is a simple yet effective post-hoc calibration method that learns a single scalar parameter to rescale logits before the softmax function, improving calibration without affecting classification accuracy.

### Method

Temperature scaling learns a single parameter *T* > 0 by minimizing the negative log-likelihood (NLL) on a held-out validation set:

```
p_calibrated = softmax(z / T)
```

where *z* are the model's raw logits. When *T* > 1, the distribution is softened (less confident); when *T* < 1, it is sharpened (more confident). The parameter is optimized using L-BFGS, a quasi-Newton method well-suited for single-parameter optimization.

### Implementation

#### Configuration (`src/stamp/modeling/config.py`)

A new `CalibrationConfig` Pydantic model is added to `TrainConfig`:

```python
class CalibrationConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 50      # L-BFGS iterations
    learning_rate: float = 0.01   # L-BFGS step size
```

- **`enabled`** (default `True`): Controls whether calibration is applied after training.
- **`max_iterations`**: Upper bound on L-BFGS optimization steps.
- **`learning_rate`**: Step size for the optimizer.

This configuration is embedded in `TrainConfig` and thereby also in `CrossvalConfig`, making it available in both single-split training and k-fold cross-validation.

#### Calibration Module (`src/stamp/modeling/calibration.py`) — *New File*

Contains:

- **`TemperatureScaler(nn.Module)`**: Wraps a trained Lightning model, adds a learnable `temperature` parameter initialized to 1.0. The `calibrate()` method collects all validation logits and targets in a single forward pass (no gradients on the base model), then optimizes temperature via L-BFGS. The temperature is clamped to [0.01, 10.0] post-optimization.

- **`calibrate_model_()`**: Entry point called from `train_model_()`. Instantiates `TemperatureScaler`, runs calibration on the validation dataloader, stores the optimal temperature in `model.hparams["temperature"]` and `model.hparams["calibrated"] = True`, then saves a separate `model_calibrated.ckpt` checkpoint.

- **`get_calibrated_probabilities()`** and **`apply_temperature_scaling()`**: Utility functions for applying a known temperature to logits during inference.

The `TemperatureScaler._get_logits_and_targets()` method handles all batch formats:
- 5-element batches: tile-level with sample weights `(bags, coords, bag_sizes, targets, weights)`
- 3-element batches: slide/patient-level with sample weights `(feats, targets, weights)`
- 4-element and 2-element legacy formats are also supported for backward compatibility.

#### Training Integration (`src/stamp/modeling/train.py`)

After training completes and the best model is reloaded from checkpoint, `train_model_()` invokes calibration:

```python
if calibration_config.enabled and task == "classification":
    calibrate_model_(
        model=best_model,
        valid_dl=valid_dl,
        output_dir=output_dir,
        config=calibration_config,
    )
```

This produces two checkpoint files:
- `model.ckpt` — the original best model (uncalibrated)
- `model_calibrated.ckpt` — the same model with `temperature` stored in hyperparameters

Both checkpoints are available for comparative evaluation.

#### Cross-Validation Integration (`src/stamp/modeling/crossval.py`)

The `calibration_config` is forwarded from `CrossvalConfig` (which inherits from `TrainConfig`) through to `train_model_()` for each fold, producing calibrated checkpoints per fold.

#### Deployment Integration (`src/stamp/modeling/deploy.py`)

During inference in `_predict()`, the deployment code checks for a stored temperature:

```python
temperature = getattr(model.hparams, "temperature", None)
if temperature is not None:
    preds = get_calibrated_probabilities(logits, temperature)
else:
    preds = torch.softmax(logits, dim=1)
```

This applies to both single-target and multi-target prediction paths. When deploying with the calibrated checkpoint, predictions automatically use temperature-scaled probabilities. When deploying with the original `model.ckpt`, standard softmax is used.

#### Statistics Integration (`src/stamp/statistics/__init__.py`, `src/stamp/statistics/calibration.py`)

A new statistics module (`statistics/calibration.py`) provides:

- **`compute_ece()`**: Computes Expected Calibration Error with equal-width binning, returning per-bin accuracies, confidences, and sample counts.
- **`plot_reliability_diagram()`**: Generates a reliability diagram (calibration curve + prediction histogram) saved as SVG, showing the relationship between predicted confidence and actual accuracy.
- **`compute_calibration_metrics()`**: Returns ECE, Maximum Calibration Error (MCE), and Brier score.

These are automatically invoked during statistics computation for classification tasks via `_compute_and_save_calibration()` in `statistics/__init__.py`. For each `(ground_truth_label, true_class)` combination, the following outputs are produced:
- `reliability-diagram_{label}={class}.svg` — visual reliability diagram
- `calibration_{label}={class}.csv` — ECE, MCE, Brier score metrics

---

## Feature 2: MILAN-Weighted Training Loss

### Motivation

In Salivary Gland Cytology, reporting the Milan category is clinically significant for patient malignancy assessment. Not all patients contribute equally to model learning — patients with Milan category V and VI provide more reliable training signal due to potentially more diagnostic visual features. By weighting the training loss according to Milan categories, the model can focus learning on the most informative samples while reducing the influence of ambiguous cases.

Critically, only the **training loss** is weighted. Validation and test losses remain unweighted to ensure that model selection and evaluation reflect true generalization performance without bias.

### Method

Each patient is assigned a scalar weight *w* ∈ (0, 1] based on their Milan category. The per-sample cross-entropy loss is computed without reduction, then multiplied by the sample weight before averaging:

```
L_weighted = mean(w_i · CE(ŷ_i, y_i))
```

where *w_i* is the MILAN confidence weight for patient *i*.

### MILAN Confidence Mapping

The mapping from Milan category to confidence weights is defined in `src/stamp/modeling/train.py`:

```python
MILAN_CONFIDENCE: dict[str, float] = {
    "I":   0.01,
    "II":  0.05,
    "III": 0.2,
    "IVb": 0.5,
    "IVa": 0.8,
    "V":   0.95,
    "VI":  1.0,
}
```

Higher Milan codes (indicating clearer classification) receive higher weights. Patients with unknown or unmapped MILAN codes default to weight 1.0.

### Implementation

#### Configuration (`src/stamp/modeling/config.py`)

A new optional field is added to `TrainConfig`:

```python
milan_table: Path | None = Field(
    default=None,
    description="Optional CSV mapping PATIENT to Milan category for sample-level loss weighting.",
)
```

When `None` (default), no weighting is applied and all behavior is unchanged from the baseline pipeline.

#### CSV Format

The MILAN table is a semicolon-separated CSV with at least two columns:
- **`PATIENT`** (or the configured `patient_label`): Patient identifier matching the clinical table.
- **`Milan-C`**: The Milan category code (e.g., "I", "II", ..., "VI").

#### Weight Loading (`src/stamp/modeling/train.py`)

`_load_milan_weights()` reads the CSV, maps each patient's Milan category to a confidence float using the `MILAN_CONFIDENCE` dictionary, and returns a `dict[PatientId, float]`. Unrecognized codes are logged as warnings and default to weight 1.0.

#### Data Layer (`src/stamp/modeling/data.py`)

The `PatientData` dataclass gains a `sample_weight: float = 1.0` field. Both dataset classes are updated:

- **`BagDataset`** (tile-level): Accepts an optional `sample_weights` tensor. `__getitem__()` returns a 5-element tuple `(bag, coords, bag_size, target, weight)`.
- **`PatientFeatureDataset`** (slide/patient-level): Accepts an optional `sample_weights` tensor. `__getitem__()` returns a 3-element tuple `(features, label, weight)`.
- **`_collate_to_tuple()`**: Updated to unpack and collate the 5th weight element, producing `(bags, coords, bag_sizes, targets, weights)` batches.

Weight tensors are constructed at dataloader creation time from `PatientData.sample_weight` values.

#### Training Integration (`src/stamp/modeling/train.py`)

In `train_categorical_model_()`, if `config.milan_table` is provided, MILAN weights are loaded and assigned to each patient's `PatientData.sample_weight` before dataloader construction. Matching statistics are logged (matched/unmatched patient counts).

#### Cross-Validation Integration (`src/stamp/modeling/crossval.py`)

The same weight-loading and assignment logic is applied in `categorical_crossval_()`, ensuring MILAN weights are carried through all k-fold splits.

#### Model Layer (`src/stamp/modeling/models/__init__.py`)

All classification model wrappers are updated to unpack the additional weight element:

- **`LitTileClassifier._step()`**: Unpacks `(bags, coords, bag_sizes, targets, sample_weights)`. During training, computes per-sample cross-entropy with `reduction='none'`, multiplies by `sample_weights`, then takes the mean. During validation/test, uses standard mean-reduction cross-entropy (weights ignored).

- **`LitSlideClassifier._step()`**: Unpacks `(feats, targets, sample_weights)`. Same training-only weighting logic.

- **Batch type annotations** are updated throughout (`training_step`, `validation_step`, `test_step`, `predict_step`) to reflect the additional tensor element.

#### Barspoon/Trans-MIL Model (`src/stamp/modeling/models/barspoon.py`)

`LitMilClassificationMixin.step()` is updated similarly:
- Unpacks `(feats, coords, bag_sizes, targets, sample_weights)`.
- During training (`step_name == "train"`): sums per-target weighted losses.
- During validation/test: uses standard unweighted cross-entropy.
- `predict_step()` updated to unpack 5-element batches.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/stamp/modeling/config.py` | Added `CalibrationConfig` model; added `calibration` and `milan_table` fields to `TrainConfig` |
| `src/stamp/modeling/train.py` | Added `_load_milan_weights()`, MILAN weight application, calibration invocation in `train_model_()`, monitor metric changed to `validation_auroc` |
| `src/stamp/modeling/crossval.py` | Added MILAN weight loading/application, calibration config forwarding, updated batch unpacking |
| `src/stamp/modeling/deploy.py` | Added temperature-aware inference in `_predict()` for both single and multi-target paths |
| `src/stamp/modeling/data.py` | Added `sample_weight` to `PatientData`, updated `BagDataset`, `PatientFeatureDataset`, `_collate_to_tuple`, `create_dataloader`, `tile_bag_dataloader` |
| `src/stamp/modeling/models/__init__.py` | Updated all classification model `_step()` methods for weighted loss; updated batch type annotations; updated `LitTileClassifier.forward()` signature |
| `src/stamp/modeling/models/barspoon.py` | Updated `LitMilClassificationMixin.step()` and `predict_step()` for weighted loss and 5-element batches |
| `src/stamp/statistics/__init__.py` | Added `_compute_and_save_calibration()`, integrated reliability diagram generation into classification stats |
| `.gitignore` | Added `stamp_bak`, `stamp_bak2` entries |

## Files Added

| File | Purpose |
|------|---------|
| `src/stamp/modeling/calibration.py` | Temperature scaling implementation: `TemperatureScaler`, `calibrate_model_()`, utility functions |
| `src/stamp/statistics/calibration.py` | Calibration metrics: `compute_ece()`, `plot_reliability_diagram()`, `compute_calibration_metrics()` |

---

## Architectural Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Configuration Layer                          │
│  TrainConfig ─── CalibrationConfig (enabled, max_iter, lr)          │
│              └── milan_table: Path | None                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         ▼                     ▼                      ▼
   train_categorical_    categorical_          deploy_categorical_
   model_()              crossval_()           model_()
         │                     │                      │
         │  ┌──────────────────┘                      │
         │  │  MILAN weights loaded                   │
         │  │  → PatientData.sample_weight            │
         ▼  ▼                                         ▼
   ┌─────────────┐                             ┌──────────────┐
   │  Dataloaders│                             │   _predict() │
   │  (5-elem or │                             │  checks for  │
   │   3-elem    │                             │  temperature │
   │   batches)  │                             │  in hparams  │
   └──────┬──────┘                             └──────┬───────┘
          │                                           │
          ▼                                           ▼
   ┌──────────────┐                            ┌──────────────┐
   │  Model _step │                            │  softmax or  │
   │  training:   │                            │  calibrated  │
   │  weighted CE │                            │  softmax     │
   │  val/test:   │                            └──────────────┘
   │  standard CE │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐         ┌─────────────────────┐
   │ train_model_ │───────> │  calibrate_model_() │
   │ (best ckpt)  │         │  TemperatureScaler  │
   └──────┬───────┘         │  → model_calibrated │
          │                 │    .ckpt            │
          │                 └─────────────────────┘
          ▼
   ┌──────────────┐
   │ Statistics   │
   │ compute_     │──▶ reliability diagram (SVG)
   │ stats_()     │──▶ calibration metrics (CSV)
   └──────────────┘    (ECE, MCE, Brier)
```

---

## Detailed Change Log

### `src/stamp/modeling/config.py`

- **Added** `CalibrationConfig` — Pydantic model with `enabled`, `max_iterations`, `learning_rate` fields, using `ConfigDict(extra="forbid")` for strict validation.
- **Added** `TrainConfig.calibration` — `CalibrationConfig` field with `default_factory` for default-enabled calibration.
- **Added** `TrainConfig.milan_table` — Optional `Path` field for MILAN weight CSV.

### `src/stamp/modeling/calibration.py` *(new)*

- **`TemperatureScaler`**: nn.Module wrapping a trained model with a single learnable temperature parameter. Supports forward pass with temperature division, and a `calibrate()` method using L-BFGS on validation NLL.
- **`calibrate_model_()`**: Orchestrates calibration — creates scaler, optimizes temperature, stores in model hyperparameters, saves calibrated checkpoint.
- **`apply_temperature_scaling()`** / **`get_calibrated_probabilities()`**: Inference-time utilities.

### `src/stamp/modeling/train.py`

- **Added** `MILAN_CONFIDENCE` dictionary mapping Milan codes to confidence weights.
- **Added** `_load_milan_weights()` function for CSV parsing and weight mapping.
- **Modified** `train_categorical_model_()` to load and assign MILAN weights when `config.milan_table` is set.
- **Modified** `train_model_()`:
  - Accepts `calibration_config` parameter.
  - Changed monitoring metric to `validation_auroc` (mode `max`) for classification tasks, replacing `validation_loss` (mode `min`).
  - After loading best checkpoint, invokes `calibrate_model_()` for classification tasks.
- **Modified** `setup_dataloaders_for_training()` — batch unpacking uses `*_` splat for forward compatibility with extra tuple elements.

### `src/stamp/modeling/crossval.py`

- **Added** import of `_load_milan_weights`.
- **Added** MILAN weight loading and assignment in `categorical_crossval_()`.
- **Modified** batch unpacking to use indexed access (`batch[0]`) instead of tuple destructuring.
- **Added** `calibration_config=config.calibration` forwarding to `train_model_()`.

### `src/stamp/modeling/deploy.py`

- **Added** import of `get_calibrated_probabilities`.
- **Modified** `_predict()`: Both single-target and multi-target classification paths now check for `model.hparams.temperature` and apply calibrated softmax when present.

### `src/stamp/modeling/data.py`

- **Added** `PatientData.sample_weight: float = 1.0` field.
- **Added** `_SampleWeight` type alias.
- **Modified** `BagDataset`:
  - New `sample_weights` field (optional `Float[Tensor, "index"]`).
  - `__getitem__()` returns 5-element tuple including weight.
- **Modified** `PatientFeatureDataset`:
  - New `sample_weights` constructor parameter.
  - `__getitem__()` returns 3-element tuple `(feats, label, weight)`.
- **Modified** `_collate_to_tuple()`: Handles 5-element items, collates weights into a tensor.
- **Modified** `create_dataloader()`: Constructs `sample_weights` tensor from `PatientData` for slide/patient-level dataloaders.
- **Modified** `tile_bag_dataloader()`: Constructs and passes `sample_weights` to `BagDataset`.

### `src/stamp/modeling/models/__init__.py`

- **Modified** `LitTileClassifier`:
  - `forward()` signature extended with `coords` and `mask` keyword arguments.
  - `_step()` unpacks 5-element batches; applies per-sample weighted CE during training only.
  - All step methods updated for 5-element batch type annotations.
  - `predict_step()` unpacks 5-element batches (ignores weights).
- **Modified** `LitSlideClassifier`:
  - `_step()` unpacks 3-element batches; applies per-sample weighted CE during training only.
  - All step methods updated for 3-element batch type annotations.
  - `predict_step()` unpacks 3-element batches.
- **Modified** version compatibility threshold from `2.5.0` to `2.4.0`.

### `src/stamp/modeling/models/barspoon.py`

- **Modified** `LitMilClassificationMixin.step()`:
  - Unpacks 5-element batches including `sample_weights`.
  - Training: per-target per-sample weighted CE loss.
  - Validation/test: standard unweighted CE loss.
- **Modified** `predict_step()`: Unpacks 5-element batches.

### `src/stamp/statistics/calibration.py` *(new)*

- **`compute_ece()`**: Equal-width binning ECE with per-bin accuracy, confidence, and count arrays.
- **`plot_reliability_diagram()`**: Matplotlib reliability diagram with calibration curve, diagonal reference, prediction histogram, and ECE annotation. Saved as SVG.
- **`compute_calibration_metrics()`**: Returns dict with ECE, MCE, and Brier score.

### `src/stamp/statistics/__init__.py`

- **Added** `_compute_and_save_calibration()`: Collects probabilities from prediction DataFrames, computes calibration metrics, generates reliability diagram, saves metrics CSV.
- **Modified** `compute_stats_()`: Calls `_compute_and_save_calibration()` for each `(ground_truth_label, true_class)` in single-target classification statistics.

---

## Scope and Limitations

- **Target task**: These changes are designed for and tested with **classification** (binary and multi-class). Regression and survival model paths are not updated for the new batch tuple format and are outside the scope of this work.
- **Supported model architectures**: ViT, MLP, and Trans-MIL for single-target classification. The barspoon multi-target model is updated for batch format compatibility but is not the focus of this work.
- **MILAN weight scale**: The confidence mapping is specifically tuned for the clinical study context and is defined as a module-level constant rather than a configurable mapping.
- **Calibration scope**: Temperature scaling is a post-hoc method that does not modify model weights — only a single scalar parameter is learned. The original (uncalibrated) checkpoint is always preserved for comparative evaluation.
- **Monitoring metric**: The early stopping and checkpoint selection metric is set to `validation_auroc` (maximize) for classification tasks, replacing the original `validation_loss` (minimize). This choice is specific to the experimental evaluation in this thesis.

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* In International Conference on Machine Learning (pp. 1321–1330). PMLR.