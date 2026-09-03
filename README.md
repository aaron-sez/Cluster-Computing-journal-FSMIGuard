# FSMIGuard: Enterprise Host-Level API Sequence Monitoring & GMM Framework

This repository provides the core architectural implementation, GMM thresholding logic, and testbed evaluation scripts supporting our manuscript.

## Repository Structure
- `config.yaml`: Testbed hyperparameters, node distributions ($N=625$), and Transformer architectural sizing.
- `fsmiguard_model.py`: PyTorch definitions for the 6-layer Transformer encoder and Multivariate GMM anomaly detection engine.
- `evaluate_testbed.py`: End-to-end execution script simulating multi-host telemetry parsing and validation.

## Reproducibility Requirements
1. Install dependencies: `pip install torch scikit-learn pyyaml numpy`
2. Run evaluation suite: `python evaluate_testbed.py`
