# SHAP-Based Explainable AI Framework for Autism Severity Classification

This repository contains the complete implementation of a machine learning framework for classifying Autism Spectrum Disorder (ASD) severity levels using 3D motor movement data captured with Microsoft Kinect V2 sensor. The system achieves **86.4% accuracy** with **100% detection rate for severe ASD cases**.

## Repository Contents

This repository includes two main components:

### 1. `autism_severity_classification_github.zip`
Complete codebase with trained models, visualization scripts, and analysis tools.

### 2. `Processeddataset.zip`
Preprocessed dataset ready for model training and evaluation.

---

## Dataset Overview

### Dataset Composition

The dataset comprises **109 children** with motion data captured during standardized movement tasks:

- **Typical Development**: 50 children (control group)
- **Moderate ASD**: 50 children (mild-to-moderate autism)
- **Severe ASD**: 9 children (severe autism requiring high-level support)

### Feature Engineering

A total of **463 kinematic features** extracted from **25 body joint points**:

- **Joint Coordinates (75 features)**: 3D coordinates (X, Y, Z) for each of 25 joints
- **Joint Angles (~25 features)**: Angular relationships between joint pairs
- **Distance Features (~300 features)**: Euclidean distances between all joint combinations
- **Gait Parameters (~10 features)**: Velocity, stride length, stance time, swing time
- **Range of Motion (~50 features)**: ROM for each joint across X and Y axes
- **Other Features**: Spatial relationships, thresholds, and position indicators

### Data Split Strategy

The dataset is divided using **stratified sampling** to maintain class proportions:

- **Training Set**: 60% (65 samples) - 30 Typical, 30 ASD, 5 Severe
- **Internal Test Set**: 20% (22 samples) - 10 Typical, 10 ASD, 2 Severe
- **External Test Set**: 20% (22 samples) - 10 Typical, 10 ASD, 2 Severe

---

## File Structure

### `Processeddataset.zip` Contents

```
Processed dataset/
├── Final dataset.xlsx              # Original complete dataset (15 MB)
└── prepared_dataset_v2.tar.gz      # Preprocessed data ready for training (676 KB)
    ├── X_train.npy                 # Training features (65 × 463)
    ├── y_train.npy                 # Training labels (65,)
    ├── X_test.npy                  # Test features (22 × 463)
    ├── y_test.npy                  # Test labels (22,)
    ├── X_external.npy              # External test features (22 × 463)
    ├── y_external.npy              # External test labels (22,)
    ├── train_data.csv              # Training data with column names
    ├── test_data.csv               # Test data with column names
    ├── external_data.csv           # External test data with column names
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
│   ├── evaluate_external.py        # External validation script
│   └── trained_models/             # Pre-trained models
│       ├── RandomForest_model.pkl  # Random Forest (229 KB)
│       ├── XGBoost_model.pkl       # XGBoost (479 KB)
│       ├── SVM_model.pkl           # SVM (184 KB)
│       ├── DNN_model.h5            # Deep Neural Network (2.0 MB)
│       ├── scaler.pkl              # StandardScaler (12 KB)
│       └── cv_results.pkl          # Cross-validation results
│
├── results/                        # Analysis outputs
│   ├── figures/                    # All manuscript figures (300 DPI)
│   └── tables/                     # Performance tables
│
├── shap_analysis_supplementary/    # SHAP explainability analysis
│   ├── shap_values/                # SHAP values for each class
│   ├── feature_importance/         # Feature importance rankings
│   └── waterfall_plots/            # Individual prediction explanations
│
├── visualization/                  # Figure generation scripts
│   └── figure_scripts/             # All visualization scripts
│       ├── plot_data_prep_flowchart_fixed.py
│       ├── plot_architecture.py
│       ├── plot_learning_curve.py
│       ├── plot_model_comparison.py
│       ├── plot_rf_test_confusion_roc.py
│       ├── plot_rf_confusion_roc_external.py
│       ├── plot_external_test_comparison.py
│       ├── shap_analysis_typical.py
│       ├── shap_analysis_moderate_asd.py
│       ├── shap_analysis_severe_asd.py
│       └── create_combined_shap_figure.py
│
└── requirements.txt                # Python dependencies
```

---

## Getting Started

### Installation

1. **Extract the datasets**:
```bash
unzip Processeddataset.zip
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

# Load test data
X_test = np.load('prepared_dataset_v2/X_test.npy')
y_test = np.load('prepared_dataset_v2/y_test.npy')

# Load external test data
X_external = np.load('prepared_dataset_v2/X_external.npy')
y_external = np.load('prepared_dataset_v2/y_external.npy')

print(f"Training: {X_train.shape}")    # (65, 463)
print(f"Test: {X_test.shape}")         # (22, 463)
print(f"External: {X_external.shape}") # (22, 463)
```

### 2. Load Data with Feature Names (CSV)

```python
import pandas as pd

# Load CSV files with column names
df_train = pd.read_csv('prepared_dataset_v2/train_data.csv')
df_test = pd.read_csv('prepared_dataset_v2/test_data.csv')
df_external = pd.read_csv('prepared_dataset_v2/external_data.csv')

# View feature names
print(df_train.columns[:10])
# ['Midspine_X', 'Midspine_Y', 'Midspine_Z', 'AnkleLeft_X', ...]

# Extract features and labels
X_train = df_train.iloc[:, :-3].values  # Exclude last 3 columns: label, label_name, id
y_train = df_train['label'].values
```

### 3. Train Random Forest Model

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
X_train = np.load('prepared_dataset_v2/X_train.npy')
y_train = np.load('prepared_dataset_v2/y_train.npy')
X_test = np.load('prepared_dataset_v2/X_test.npy')
y_test = np.load('prepared_dataset_v2/y_test.npy')

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with class weighting
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight={0: 0.72, 1: 0.72, 2: 4.33},  # Balanced weights (auto-computed)
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, 
                          target_names=['Typical', 'ASD', 'Severe']))
```

### 4. Use Pre-trained Models

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

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

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

### 5. Generate SHAP Explanations

```python
import shap
import pickle
import numpy as np

# Load model and data
with open('github_package/models/trained_models/RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = np.load('prepared_dataset_v2/X_test.npy')

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, 
                 feature_names=column_names,
                 class_names=['Typical', 'ASD', 'Severe'])
```

### 6. Generate Manuscript Figures

```python
# Navigate to visualization scripts
cd github_package/visualization/figure_scripts

# Generate all figures
python plot_data_prep_flowchart_fixed.py
python plot_architecture.py
python plot_learning_curve.py
python plot_model_comparison.py
python plot_rf_test_confusion_roc.py
python plot_rf_confusion_roc_external.py
python plot_external_test_comparison.py
python shap_analysis_typical.py
python shap_analysis_moderate_asd.py
python shap_analysis_severe_asd.py
```

---

## Model Performance

### Random Forest (Recommended)

| Metric | Internal Test | External Test |
|--------|--------------|---------------|
| **Overall Accuracy** | 86.4% | 86.4% |
| **Typical F1-Score** | 0.84 | 0.84 |
| **Moderate ASD F1-Score** | 0.86 | 0.86 |
| **Severe ASD F1-Score** | 1.00 | 1.00 |
| **Severe ASD Detection** | 2/2 (100%) | 2/2 (100%) |
| **Training Time** | 0.33 seconds | - |

### Comparison with Other Algorithms

| Algorithm | Test Accuracy | Severe ASD Detection (Internal) | Severe ASD Detection (External) |
|-----------|--------------|--------------------------------|--------------------------------|
| **Random Forest** | 86.4% | 2/2 (100%) | 2/2 (100%) |
| **XGBoost** | 81.8% | 0/2 (0%) | 0/2 (0%) |
| **SVM** | 68.2% | 0/2 (0%) | 0/2 (0%) |

**Key Finding**: Only Random Forest successfully detected all severe ASD cases in both test sets, demonstrating superior clinical reliability.

---

## Clinical Significance

### Perfect Severe ASD Detection

The model's **100% accuracy** in identifying severe ASD cases represents a critical breakthrough for clinical applications:

- **Early Intervention**: Timely detection enables immediate support for children requiring high-level assistance
- **Clinical Trust**: Consistent performance across validation sets builds confidence for real-world deployment
- **Objective Assessment**: Non-invasive motor analysis provides quantitative measures complementing behavioral assessments

### Top Motor Biomarkers (SHAP Analysis)

**Typical Development**:
- WristRight_Y (0.0121)
- ElRTFoR_X (0.0097)
- KneeRight_Y (0.0097)

**Moderate ASD**:
- FoRTKeR_X (0.0110)
- KneeRight_Y (0.0102)
- FoLTShR_X (0.0102)

**Severe ASD**:
- ElLTFoL_X (0.0181)
- FoRTKeR_X (0.0142)
- RomHaRx_X (0.0140)

---

**Workflow**:
1. Train model on training set
2. Tune hyperparameters using internal test set
3. Select best parameters
4. Retrain on combined (train + test) data
5. **ONLY AT THE END**: Evaluate on external test set

### Class Imbalance Handling

Due to severe class imbalance (5 severe vs. 30 typical/moderate in training):

```python
# Class weights computed using sklearn's 'balanced' mode
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
# Result: {0: 0.72, 1: 0.72, 2: 4.33}
```

### Feature Names

All 463 features are documented with meaningful names:
- Joint coordinates: `{JointName}_{X/Y/Z}`
- Distances: `{Joint1}T{Joint2}_{X/Y/Z}`
- Angles: Abbreviated joint pair names
- ROM: `Rom{JointName}{x/y}`
- Gait: `Velocity`, `StrLe`, `GaCT`, etc.

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
Fırat, Y. (2025). SHAP-Based Explainable AI Framework for Autism 
Severity Classification Using 3D Motor Biomarkers. 
[Journal Name], [Volume](Issue), pp.xx-xx.
```

---

## Contact

For questions, issues, or collaboration opportunities, please contact:

**Yelda Fırat**  
Mudanya University  
Email: yelda.firat@mudanya.edu.tr

---

## Acknowledgments

This research builds upon the foundational work of Al-Jubouri et al. in collecting and sharing the Kinect-based autism movement dataset. We acknowledge their contribution to advancing autism research through open data sharing.

