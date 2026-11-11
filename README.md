# SmartIrrigation-Vision 🍅💧

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.27%25-brightgreen?style=for-the-badge&logo=target)](https://github.com/rohitt2008/SmartIrrigation-Vision)
[![5-Fold CV](https://img.shields.io/badge/5--Fold%20CV-98.75%25%20%C2%B1%200.73%25-success?style=for-the-badge)](https://github.com/rohitt2008/SmartIrrigation-Vision)
[![Parameters](https://img.shields.io/badge/Parameters-1.24M%20(94.7%25%20Reduction)-blue?style=for-the-badge)](https://github.com/rohitt2008/SmartIrrigation-Vision)
[![Latency](https://img.shields.io/badge/Inference-1.66%20ms%20(CPU)-orange?style=for-the-badge)](https://github.com/rohitt2008/SmartIrrigation-Vision)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Compact Residual Convolutional Neural Network for Multi-Stage Tomato Leaf Water Stress Identification and Explainability**

---

## 📌 Project Overview

Precision irrigation scheduling is vital for sustainable agricultural water management, mitigating drought stress, and optimizing crop yield. Traditional soil moisture probes provide localized point data and are maintenance-intensive, while aerial hyperspectral imaging involves significant operational expense.

**SmartIrrigation-Vision** provides a non-destructive computer vision solution that identifies four discrete irrigation stress regimes directly from tomato (*Solanum lycopersicum*) leaf imagery. Utilizing **LightResNet**, a compact 10-layer residual convolutional neural network with custom $1 \times 1$ skip-connection projections, the architecture achieves **99.27% Test Accuracy** and **0.9902 Cohen's Kappa** while requiring only **1.24 Million parameters** (~4.98 MB model size) and **0.109 GFLOPs**, with an average CPU inference latency of **1.66 ms per image**.

Integrated with **Gradient-weighted Class Activation Mapping (Grad-CAM)**, the system enables visual inspection of internal attention patterns across different hydration stages.

---

## 🌟 Key Highlights & Technical Specifications

* ⚡ **Compact Architecture**: Custom 10-layer residual CNN with **1,240,948 parameters** and **0.109 GFLOPs**, representing a **94.7% parameter reduction** compared to standard ResNet-50 (23.59M).
* ⏱️ **Edge-Viable Inference**: Measured CPU inference latency of **1.66 ms per image** (~600 FPS throughput) without requiring post-training quantization.
* 🎯 **Strong Classification Performance**: **99.27% accuracy** (6,241 / 6,287 correct) on the independent test set, validated across a **5-fold cross-validation scheme (98.75% ± 0.73%)**.
* 📏 **Ordinal Metric Verification**: Mean Absolute Error (MAE) of **0.0073 class levels** and Quadratic Weighted Kappa of **0.9935**, confirming that misclassifications are confined to adjacent moisture levels.
* 🛡️ **Stratified Evaluation**: Strict 70/30 train/test split on 20,955 images with an isolated 10% validation subset.
* 👁️ **Explainability Analysis (Grad-CAM)**: Regional attention maps highlighting foliar features in drought-stressed classes while identifying contour attention in hydrated classes.

---

## 🏗 Architecture: LightResNet

Standard deep networks such as ResNet-50 or VGG-16 are over-parameterized for individual leaf classification on uniform backgrounds. Our **LightResNet** architecture streamlines residual feature extraction by:

1. Employing four compact residual blocks (`Block-B` through `Block-E`) with channel progressions of $16 \to 32 \to 64 \to 128 \to 256$.
2. Implementing dual $3 \times 3$ convolutions in the main path paired with dimension-matching $1 \times 1$ convolutions in the shortcut projection path.
3. Incorporating Batch Normalization and ReLU non-linearities across all convolutional and dense layers.

```
Input Leaf Image (224 x 224 x 3)
         │
  [Block-A: Conv 3x3 s=2 -> BN -> ReLU -> Conv 3x3 s=2 -> BN -> MaxPool 2x2] ──► (28 x 28 x 16)
         │
  [Block-B: Residual Block (16 -> 32) + MaxPool 2x2] ──► (14 x 14 x 32)
         │
  [Block-C: Residual Block (32 -> 64) + MaxPool 2x2] ──► (7 x 7 x 64)
         │
  [Block-D: Residual Block (64 -> 128) + MaxPool 2x2] ──► (3 x 3 x 128)
         │
  [Block-E: Residual Block (128 -> 256) + MaxPool 3x3] ──► (1 x 1 x 256)
         │
  [Flatten] ──► 256-dim Feature Vector
         │
  [Linear (256 -> 64) + ReLU]
         │
  [Linear (64 -> 4)] ──► Class Logits [LI, MI, HI, FI]
```

### Layer-by-Layer Parameters

| Stage | Operations | Output Shape | Parameters |
| :--- | :--- | :---: | :---: |
| **Input** | RGB Leaf Image | $224 \times 224 \times 3$ | 0 |
| **Block-A** | Conv2D ($3\times3$, s=2) + BN + Conv2D ($3\times3$, s=2) + BN + MaxPool | $28 \times 28 \times 16$ | 2,800 |
| **Block-B** | Residual Block ($16 \to 32$) + MaxPool | $14 \times 14 \times 32$ | 14,528 |
| **Block-C** | Residual Block ($32 \to 64$) + MaxPool | $7 \times 7 \times 64$ | 57,664 |
| **Block-D** | Residual Block ($64 \to 128$) + MaxPool | $3 \times 3 \times 128$ | 230,144 |
| **Block-E** | Residual Block ($128 \to 256$) + MaxPool | $1 \times 1 \times 256$ | 919,040 |
| **Classifier** | Linear ($256 \to 64$) + ReLU + Linear ($64 \to 4$) | 4 | 16,708 |
| **Total** | **All Trainable Parameters** | — | **1,240,948** |

---

## 📊 Dataset & Irrigation Regimes

The experiments are conducted on the standardized *"Irrigation Management in Tomato Plants"* dataset containing **20,955 high-resolution leaf images** captured across controlled soil moisture regimes.

![Dataset Distribution](dataset_distribution.png)
*Figure 1: Class distribution across the four irrigation categories (20,955 total samples).*

### Class Definitions & Sample Counts

| Class Code | Category | Soil Field Capacity (FC) | Morphological Characteristics | Total Images | Train Set (70%) | Test Set (30%) |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| **LI** | Low Irrigated | **25% FC** | Severe moisture deficit, blade rolling, turgor loss, chlorosis | 5,247 | 3,673 | 1,574 |
| **MI** | Medium Irrigated | **50% FC** | Moderate deficit, early structural curling, slight drooping | 5,170 | 3,619 | 1,551 |
| **HI** | Highly Irrigated | **75% FC** | Near-optimal hydration, upright posture, vibrant green pigmentation | 5,108 | 3,575 | 1,533 |
| **FI** | Fully Irrigated | **100% FC** | Complete field capacity saturation, expansive lamina | 5,430 | 3,801 | 1,629 |
| **TOTAL** | — | — | — | **20,955** | **14,668** | **6,287** |

---

## 📈 Training Dynamics & Convergence

Training was conducted using the Adam optimizer ($\eta = 10^{-4}$, $\beta_1 = 0.9, \beta_2 = 0.999$) with Cross-Entropy Loss and an Early Stopping patience of 3 epochs on validation loss.

![Training History](training_history.png)
*Figure 2: Training loss trajectory (left) and validation metric progression (right) across epochs.*

* Training concluded at epoch 8 when validation loss reached a plateau (~0.052).
* Total training time: ~3.5 minutes on Apple Silicon GPU (MPS) / NVIDIA CUDA.

---

## 🔬 Experimental Results & Comprehensive Evaluation

### 1. Overall Performance Metrics (Held-Out Test Set: 6,287 Images)

| Evaluation Metric | Meaning | Result |
| :--- | :--- | :---: |
| **Test Accuracy** | Fraction of correct classifications | **99.27%** (6,241 / 6,287) |
| **Macro Precision** | Unweighted average precision across 4 classes | **99.27%** |
| **Macro Recall** | Unweighted average recall (sensitivity) across 4 classes | **99.26%** |
| **Macro F1-Score** | Harmonic mean of precision and recall | **99.26%** |
| **Cohen's Kappa ($\kappa$)** | Statistical agreement corrected for chance | **0.9902** |
| **Mean Absolute Error (MAE)** | Average ordinal distance error | **0.0073 class levels** |
| **Quadratic Weighted Kappa** | Penalized ordinal agreement metric | **0.9935** |

---

### 2. Per-Class Classification Breakdown

| Class | Precision (%) | Recall (%) | F1-Score (%) | Test Support (Images) |
| :--- | :---: | :---: | :---: | :---: |
| **LI (Low Irrigated - 25%)** | 98.31% | 99.68% | 98.99% | 1,574 |
| **MI (Medium Irrigated - 50%)** | 99.09% | 98.19% | 98.64% | 1,551 |
| **HI (Highly Irrigated - 75%)** | 99.67% | 99.41% | 99.54% | 1,533 |
| **FI (Fully Irrigated - 100%)** | 100.00% | 99.75% | 99.88% | 1,629 |
| **Macro Average** | **99.27%** | **99.26%** | **99.26%** | **6,287** |

![Class Metrics](class_metrics.png)
*Figure 3: Breakdown of Precision, Recall, and F1-Score across all four irrigation categories.*

---

### 3. Confusion Matrix Analysis

Out of 6,287 test samples, 6,241 were classified correctly. All 46 misclassifications occurred exclusively between adjacent moisture levels:

| Actual \ Predicted | Predicted: **LI** | Predicted: **MI** | Predicted: **HI** | Predicted: **FI** | Sensitivity (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual: LI (25%)** | **1,569** | 5 | 0 | 0 | **99.68%** |
| **Actual: MI (50%)** | 27 | **1,523** | 1 | 0 | **98.19%** |
| **Actual: HI (75%)** | 0 | 9 | **1,524** | 0 | **99.41%** |
| **Actual: FI (100%)** | 0 | 0 | 4 | **1,625** | **99.75%** |

![Confusion Matrix](confusion_matrix.png)
*Figure 4: Confusion matrix evaluated over 6,287 unseen test samples.*

---

### 4. 5-Fold Cross-Validation

| Validation Fold | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Cohen's Kappa ($\kappa$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 99.12% | 99.13% | 99.11% | 99.12% | 0.9883 |
| **Fold 2** | 98.45% | 98.47% | 98.44% | 98.45% | 0.9793 |
| **Fold 3** | 99.02% | 99.01% | 99.03% | 99.02% | 0.9869 |
| **Fold 4** | 97.64% | 97.66% | 97.61% | 97.63% | 0.9685 |
| **Fold 5** | 99.52% | 99.50% | 99.52% | 99.51% | 0.9936 |
| **Mean $\pm$ Std** | **98.75% $\pm$ 0.73%** | **98.75% $\pm$ 0.73%** | **98.74% $\pm$ 0.74%** | **98.74% $\pm$ 0.74%** | **0.9833 $\pm$ 0.0098** |

---

### 5. Benchmark Comparison with Published Baselines

The table below benchmarks LightResNet against baseline architectures evaluated on this same 20,955-image dataset by Shahi et al. (IEEE Transactions on Artificial Intelligence, 2025):

| Architecture | Total Parameters | GFLOPs | Accuracy (%) | Source |
| :--- | :---: | :---: | :---: | :--- |
| **MobileNetV2** | 2.26M | 0.59 | 70.52% | Shahi et al. (2025) |
| **NASNet-Mobile** | 4.23M | 1.13 | 70.10% | Shahi et al. (2025) |
| **MobileNet** | 3.23M | 1.14 | 82.52% | Shahi et al. (2025) |
| **DenseNet-121** | 6.95M | 5.67 | 84.07% | Shahi et al. (2025) |
| **VGG-16** | 14.71M | 30.70 | 84.98% | Shahi et al. (2025) |
| **ResNet-50** | 23.59M | 7.72 | 91.93% | Shahi et al. (2025) |
| **Shahi et al. CNN** | 1.24M | 0.11 | 98.90% | Shahi et al. (2025) |
| **LightResNet (Ours)** | **1.24M** | **0.109** | **99.27%** | This Study |

---

## 🧠 Explainable AI: Grad-CAM Saliency Analysis

Grad-CAM was applied to the penultimate convolutional layer (`Block-E`):

![Grad-CAM Visualization](gradcam_results.png)
*Figure 5: Grad-CAM attention heatmaps across the four irrigation categories. Warm colors indicate highest gradient activation.*

### Analysis & Interpretability Nuance

* **Low Irrigated (LI)**: Heatmaps concentrate on the **curled leaf margins and blade tips**, reflecting localized loss of hydrostatic cell turgor and leaf rolling.
* **Medium Irrigated (MI)**: Attention focuses along the **central leaf blade and primary veins**, capturing early structural moisture reduction.
* **HI & FI (Hydrated Categories)**: Attention spreads across the expanded lamina. In the absence of localized stress lesions, some activation extends to the perimeter background boundary.
* **Agronomic Consideration**: Foliar chlorosis can also be induced by nutrient deficiencies (e.g., nitrogen) or viral infection; therefore, vision models in unconstrained field environments should be paired with ambient telemetry (temperature, VPD) to ensure holistic diagnostic accuracy.

---

## 💻 Quick Start & Reproducibility

### 1. Setup Environment

```bash
git clone https://github.com/rohitt2008/SmartIrrigation-Vision.git
cd SmartIrrigation-Vision

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Model Evaluation

To evaluate the saved `best_model.pth` checkpoint and regenerate plots:

```bash
python3 evaluate.py
```

### 3. Generate Grad-CAM Heatmaps

```bash
python3 gradcam_visualizer.py
```

### 4. Training

To train the network from scratch:

```bash
python3 train.py --epochs 40 --batch_size 64 --lr 0.0001
```

---

## 📜 Citations & Academic References

1. **Dataset & Benchmark Article**:
   > T. B. Shahi, C. Sitaula, K. P. Bhandari, S. Poudel, R. Bhandari, R. Mishra, B. K. Sharma, and B. Mishra, *"Deep Learning-Based Method for Irrigation Status Detection in Tomato Using Plant Leaves,"* **IEEE Transactions on Artificial Intelligence**, vol. 6, no. 7, pp. 1849–1858, July 2025. DOI: [10.1109/TAI.2024.3377759](https://doi.org/10.1109/TAI.2024.3377759).

2. **Dataset Repository**:
   > *"Irrigation Management in Tomato Plants Dataset,"* Figshare Repository. DOI: [10.6084/m9.figshare.22297915](https://doi.org/10.6084/m9.figshare.22297915).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
