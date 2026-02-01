# SHAP-Based Explainable AI Framework for Autism Severity Classification Using 3D Motor Biomarkers

This repository contains the complete implementation of a machine learning framework for classifying Autism Spectrum Disorder (ASD) severity levels using 3D motor movement data captured with Microsoft Kinect V2 sensor. The system achieves **86.4% accuracy** for typical and moderate ASD classes, with **100% detection rate for severe ASD cases on synthetic data**.

> **Important Note**: Severe ASD classification results are based on synthetically generated features and should be interpreted as methodological proof-of-concept rather than clinical validation. The model has not been validated on real Kinect-derived severe ASD motor data.

## Repository Contents

This repository includes two main components:

### 1. `autism_severity_classification_github.zip`
Complete codebase with trained models, visualization scripts, and analysis tools.

### 2. `Processed dataset.zip`
Preprocessed dataset ready for model training and evaluation.

---

## Dataset Overview

### Dataset Composition

The dataset comprises **109 children** with motion data captured during standardized movement tasks:

- **Typical Development**: 50 children (control group)
- **Moderate ASD**: 50 children (mild-to-moderate autism)
- **Severe ASD**: 9 children (severe autism - **synthetic features generated from moderate ASD data**)

> **Note on Severe ASD Data**: The original dataset contains only 2D video recordings for severe ASD cases without Kinect V2 3D joint coordinates. Synthetic features were created by averaging moderate ASD vectors and adding Gaussian noise (μ=0, σ=0.15) to maintain the 463-feature structure for consistent comparisons.

### Feature Engineering

A total of **463 kinematic features** extracted from **25 body joint points**:

- **Joint Coordinates (75 features)**: 3D coordinates (X, Y, Z) for each of 25 joints
- **Distance Features (~300 features)**: Euclidean distances between joint pairs (e.g., ElLTFoL_X: left elbow to left foot X-axis distance)
- **Range of Motion (~50 features)**: ROM for each joint across X and Y axes (e.g., RomHaRx_X: right hand X-axis ROM)
- **Gait Parameters (~10 features)**: Velocity, stride length, stance time, swing time
- **Other Features**: Spatial relationships and position indicators

### Data Split Strategy

The dataset is divided using **stratified sampling** to maintain class proportions:

| Set | Samples | Typical | Moderate ASD | Severe ASD | Purpose |
|-----|---------|---------|--------------|------------|---------|
| **Training** | 65 (60%) | 30 | 30 | 5 | Model training |
| **Internal Test** | 22 (20%) | 10 | 10 | 2 | Interim evaluation during development |
| **Held-Out Test** | 22 (20%) | 10 | 10 | 2 | Final validation (never accessed during training) |

> **Note**: Both test sets were drawn from the same original dataset using the same sensor, protocol, and feature extraction pipeline, representing reserved subsets from the same data distribution rather than independent external validation.

---

## File Structure

### `Processed dataset.zip` Contents

```
Processed dataset/
├── Final dataset.xlsx              # Original complete dataset (15 MB)
└── prepared_dataset_v2.tar.gz      # Preprocessed data ready for training (676 KB)
    ├── X_train.npy                 # Training features (65 × 463)
    ├── y_train.npy                 # Training labels (65,)
    ├── X_test.npy                  # Internal test features (22 × 463)
    ├── y_test.npy                  # Internal test labels (22,)
    ├── X_external.npy              # Held-out test features (22 × 463)
    ├── y_external.npy              # Held-out test labels (22,)
    ├── train_data.csv              # Training data with column names
    ├── test_data.csv               # Internal test data with column names
    ├── external_data.csv           # Held-out test data with column names
    ├── column_names.txt            # List of 463 feature names
    ├── metadata.pkl                # Python metadata dictionary
    └── README.txt                  # Dataset documentation
```

### `autism_severity_classification_github.zip` Contents

```
github_package/
├── PERFORMANCE_TABLES/             # Performance metrics in CSV format
│   ├── CLASSWISE_PERFORMANCE.csv
│   ├── CONFUSION_MATRIX_EXTERNAL.csv
│   ├── CONFUSION_MATRIX_TEST.csv
│   ├── MODEL_COMPARISON_TABLE.csv
│   ├── OVERALL_PERFORMANCE.csv
│   └── PERFORMANCE_METRICS_DETAILED.csv
│
├── data/                           # Data preparation scripts
│   ├── prepare_dataset_v2.py       # Dataset preprocessing pipeline
│   └── validate_dataset.py         # Data validation utilities
│
├── docs/                           # Documentation
│   ├── CLINICAL_GUIDE.md           # Clinical usage guidelines
│   ├── DATA_STRUCTURE.md           # Data structure documentation
│   └── METHODOLOGY.md              # Methodology overview
│
├── inference/                      # Prediction and deployment
│   ├── predict.py                  # Prediction script for new data
│   └── clinical_use_example.py     # Clinical deployment example
│
├── models/                         # Model training and evaluation
│   ├── train_model.py              # Complete training pipeline
│   ├── evaluate_external.py        # Held-out test validation script
│   └── trained_models/             # Pre-trained models
│       ├── RandomForest_model.pkl  # Random Forest (229 KB) - Recommended
│       ├── XGBoost_model.pkl       # XGBoost (479 KB)
│       ├── SVM_model.pkl           # SVM (184 KB)
│       ├── DNN_model.h5            # Deep Neural Network (2.0 MB)
│       ├── scaler.pkl              # StandardScaler (12 KB)
│       └── cv_results.pkl          # Cross-validation results
│
├── results/                        # Analysis outputs
│   ├── figures/                    # All manuscript figures (300 DPI, 12 figures)
│   └── tables/                     # Performance tables (4 tables)
│
├── shap_analysis_supplementary/    # SHAP explainability analysis
│   ├── typicial_supplementary_materials_shap_analysis/
│   │   └── typical_feature_importance_shap.csv
│   ├── asd_moderate_supplementary_materials_shap_analysis/
│   │   └── moderate_asd_feature_importance_shap.csv
│   └── asd_severe_supplementary_materials_shap_analysis/
│       └── severe_asd_feature_importance_shap.csv
│
├── visualization/                  # Figure generation scripts
│   └── figure_scripts/             # All visualization scripts
│       ├── plot_data_prep_flowchart_fixed.py
│       ├── plot_data_distribution.py
│       ├── plot_skeleton_biomarkers.py      # Figure 3: Skeleton + biomarkers
│       ├── plot_architecture.py
│       ├── plot_learning_curve.py
│       ├── plot_model_comparison.py
│       ├── plot_rf_test_confusion_roc.py
│       ├── rf_confusion_roc_heldout.py
│       ├── plot_heldout_test_comparison.py  # Figure 12: Model comparison
│       ├── create_combined_shap_figure.py
│       └── ... (additional scripts)
│
└── requirements.txt                # Python dependencies
```

---

## Getting Started

### Installation

1. **Extract the datasets**:
```bash
unzip "Processed dataset.zip"
cd "Processed dataset"
tar -xzf prepared_dataset_v2.tar.gz
```

2. **Extract the codebase**:
```bash
unzip autism_severity_classification_github.zip
cd github_package
```

3. **Install dependencies** (Python 3.8+):
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap
- tensorflow (for DNN model)

---

## Usage Examples

### 1. Load Preprocessed Data (NumPy)

```python
import numpy as np

# Load training data
X_train = np.load('prepared_dataset_v2/X_train.npy')
y_train = np.load('prepared_dataset_v2/y_train.npy')

# Load internal test data
X_test = np.load('prepared_dataset_v2/X_test.npy')
y_test = np.load('prepared_dataset_v2/y_test.npy')

# Load held-out test data
X_external = np.load('prepared_dataset_v2/X_external.npy')
y_external = np.load('prepared_dataset_v2/y_external.npy')

print(f"Training: {X_train.shape}")       # (65, 463)
print(f"Internal Test: {X_test.shape}")   # (22, 463)
print(f"Held-Out Test: {X_external.shape}") # (22, 463)
```

### 2. Use Pre-trained Models

```python
import pickle
import numpy as np

# Load pre-trained Random Forest model
with open('github_package/models/trained_models/RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('github_package/models/trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load new data
X_new = np.load('your_new_data.npy')

# Preprocess and predict
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)

# Class mapping: 0=Typical, 1=Moderate ASD, 2=Severe ASD
class_names = ['Typical', 'Moderate ASD', 'Severe ASD']
print(f"Prediction: {class_names[predictions[0]]}")
```

### 3. Generate SHAP Explanations

```python
import shap
import pickle
import numpy as np

# Load model and data
with open('github_package/models/trained_models/RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = np.load('prepared_dataset_v2/X_test.npy')

# Load feature names
with open('prepared_dataset_v2/column_names.txt', 'r') as f:
    column_names = f.read().strip().split('\n')

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, 
                 feature_names=column_names,
                 class_names=['Typical', 'Moderate ASD', 'Severe ASD'])
```

---

## Model Performance

### Random Forest (Recommended)

| Metric | Internal Test | Held-Out Test |
|--------|--------------|---------------|
| **Overall Accuracy** | 86.4% | 86.4% |
| **Typical F1-Score** | 0.84 | 0.84 |
| **Moderate ASD F1-Score** | 0.86 | 0.86 |
| **Severe ASD F1-Score** | 1.00* | 1.00* |
| **Cross-Validation (5-fold)** | 84.6% ± 10.9% | - |

*Note: Severe ASD results are based on synthetic data (95% CI: 39.8%-100.0%)

### Model Comparison on Held-Out Test

| Algorithm | Accuracy | Severe ASD Detection | Training Time |
|-----------|----------|---------------------|---------------|
| **Random Forest** | 86.4% | 2/2 (100%)* | 0.33 sec |
| **XGBoost** | 81.8% | 0/2 (0%) | 0.45 sec |
| **SVM** | 68.2% | 0/2 (0%) | 0.12 sec |

*Synthetic data - methodological proof-of-concept only

**Key Finding**: Only Random Forest successfully detected all severe ASD cases, demonstrating superior ability to capture subtle motor patterns in the synthetic data.

---

## Top Motor Biomarkers (SHAP Analysis)

### Common Biomarker Across All Classes
**ElLTFoL_X** (left elbow to left foot X-axis distance) is the only feature appearing in the top-5 biomarkers for all three classes.

### Class-Specific Top-5 Biomarkers

| Rank | Typical Development | Moderate ASD | Severe ASD |
|------|---------------------|--------------|------------|
| 1 | WristRight_Y (0.0121) | FoRTKeR_X (0.0110) | ElLTFoL_X (0.0181) |
| 2 | ElRTFoR_X (0.0097) | KneeRight_Y (0.0102) | FoRTKeR_X (0.0142) |
| 3 | KneeRight_Y (0.0097) | FoLTShR_X (0.0102) | RomHaRx_X (0.0140) |
| 4 | HandTipRight_Y (0.0094) | ElLTFoL_X (0.0101) | FoLTShR_X (0.0138) |
| 5 | ElLTFoL_X (0.0083) | RomMidx_X (0.0076) | FoRTThL_X (0.0130) |

### Systematic Biomarker Patterns

| Feature Type | Typical | Moderate ASD | Severe ASD |
|--------------|---------|--------------|------------|
| Y-axis features | 30% | 10% | 0% |
| X-axis features | 70% | 90% | 100% |
| Position features | 30% | 10% | 0% |
| Distance features | 50% | 60% | 65% |
| ROM features | 15% | 25% | 35% |

---

## Validation Strategy

### Three-Stage Validation

1. **5-Fold Cross-Validation**: Performed on training data (n=65) to assess model stability
2. **Internal Test Set**: Used for interim evaluation during development (n=22)
3. **Held-Out Test Set**: Reserved exclusively for final validation, never accessed during training (n=22)

### Class Imbalance Handling

```python
# Class weights computed using sklearn's 'balanced' mode
from sklearn.utils.class_weight import compute_class_weight

class_weights = {0: 0.72, 1: 0.72, 2: 4.33}  # Severe ASD boosted
```

---

## Limitations

1. **Synthetic Severe ASD Data**: The model has not been validated on real Kinect-derived severe ASD motor data
2. **Small Sample Size**: n=109 total, with only 9 severe ASD cases (synthetic)
3. **Wide Confidence Intervals**: Severe ASD 95% CI: 39.8%-100.0% indicates high statistical uncertainty
4. **Same Distribution**: Both test sets are from the same dataset, not independent external validation

---

## Manuscript Figures

The repository includes all 12 manuscript figures at 300 DPI resolution:

| Figure | Description |
|--------|-------------|
| Figure 1 | Data preparation flowchart |
| Figure 2 | Class distribution and data split |
| Figure 3 | Kinect V2 skeleton structure and key motor biomarkers |
| Figure 4 | Random Forest model architecture |
| Figure 5 | Learning curve analysis |
| Figure 6 | Confusion matrix and ROC curves (internal test) |
| Figure 7 | Confusion matrix and ROC curves (held-out test) |
| Figure 8 | Model comparison (RF, XGBoost, SVM) |
| Figure 9 | SHAP analysis - Typical Development |
| Figure 10 | SHAP analysis - Moderate ASD |
| Figure 11 | SHAP analysis - Severe ASD |
| Figure 12 | Model performance comparison on held-out test |

---

## Citation

If you use this dataset or code in your research, please cite:

**Dataset Source**:
```
Al-Jubouri, A., Hadi, I., & Rajihy, Y. (2020).
Three dimensional dataset combining gait and full body movement
of children with autism spectrum disorders collected by Kinect v2 camera.
Zenodo. https://doi.org/10.5061/dryad.s7h44j150
```

**This Implementation**:
```
Fırat, Y. (2026). SHAP-Based Explainable AI Framework for Autism 
Severity Classification Using 3D Motor Biomarkers. 
Frontiers in Computational Neuroscience.
```

---

## Contact

For questions, issues, or collaboration opportunities, please contact:

**Yelda Fırat**  
Mudanya University, Department of Computer Engineering  
Email: yelda.firat@mudanya.edu.tr

---

## License

This project is released for academic and research purposes. Please cite the original dataset and this implementation when using the code or data.

## Acknowledgments

This research builds upon the foundational work of Al-Jubouri et al. in collecting and sharing the Kinect-based autism movement dataset. We acknowledge their contribution to advancing autism research through open data sharing.
