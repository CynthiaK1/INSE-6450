# RobustOps

CI/CD-Integrated Adversarial Robustness Testing and ML Risk Scoring  
INSE 6450 — AI in Systems Engineering | Winter 2026  
Cynthia Musila — 40311473  

---

## Milestone 1 Notebook (Primary Entry Point)

For Milestone 1, the complete end-to-end implementation is provided in:

RobustOps_Milestone1.ipynb

The notebook contains:

- ResNet-18 training on CIFAR-10  
- Implementation of FGSM and PGD adversarial attacks  
- Generation of adversarial telemetry (80,000 records)  
- Robustness feature engineering  
- Composite Robustness Risk Score computation  
- Deploy/Block gate decision logic  
- Exploratory Data Analysis (EDA) and visualization  
- Final risk report generation  

The notebook is fully self-contained and documents:

- Dataset profiling and validation  
- Model convergence behavior  
- Adversarial accuracy degradation across perturbation budgets  
- Per-class attack success rates  
- Confidence drop analysis  
- Risk score monotonicity with increasing epsilon  

For Milestone 1, the notebook should be considered the primary artifact.


## Overview

RobustOps automates adversarial robustness evaluation for ML models. It runs
model-specific attacks (FGSM, PGD), computes a composite Robustness Risk Score,
and enforces configurable deploy/block gates aligned with Canadian and U.S. AI
governance standards (DADM, AIDA, NIST RMF).

## Project Structure
```
RobustOps/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml      # All configurable parameters
├── src/
│   ├── train_model.py            # Train ResNet-18 on CIFAR-10
│   ├── adversarial_attacks.py    # FGSM and PGD implementations
│   ├── generate_telemetry.py     # Run attacks and generate telemetry CSV
│   ├── feature_engineering.py    # Compute robustness features
│   ├── risk_scoring.py           # Composite risk score + deploy gate
│   └── eda.py                    # All EDA plots and data profiling
├── data/                         # CIFAR-10 auto-downloaded here
└── outputs/                      # All outputs saved here
    ├── resnet18_cifar10.pth
    ├── adversarial_telemetry.csv
    ├── robustness_features.csv
    ├── risk_report.csv
    └── plots/
```

## Setup
```bash
pip install -r requirements.txt
```

## How to Run

Run each step sequentially from the project root:
```bash
# Step 1: Train the target model
python src/train_model.py

# Step 2: Generate adversarial telemetry
python src/generate_telemetry.py

# Step 3: Compute robustness features
python src/feature_engineering.py

# Step 4: Compute risk scores and gate decisions
python src/risk_scoring.py

# Step 5: Generate all EDA plots and data summaries
python src/eda.py
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/resnet18_cifar10.pth` | Trained model checkpoint |
| `outputs/adversarial_telemetry.csv` | Raw per-sample attack results (80K rows) |
| `outputs/robustness_features.csv` | Aggregated robustness metrics |
| `outputs/risk_report.csv` | Final risk scores with gate decisions |
| `outputs/plots/` | All visualization plots for EDA |

## Configuration

All parameters are in `configs/default_config.yaml`:
- Attack types and epsilon values
- Risk score weights
- Deploy/block threshold
- Data and model paths

## Dependencies

- Python 3.10+
- PyTorch >= 2.0
- torchvision, numpy, pandas, matplotlib, seaborn, scikit-learn, pyyaml
