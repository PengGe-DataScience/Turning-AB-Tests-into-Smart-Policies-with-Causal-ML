# Turning-AB-Tests-into-Smart-Policies-with-Causal-ML

**Disclosure:** This repository explains and extends the methodology of von Zahn et al., *“Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning”* (Marketing Science, Jul–Aug 2025) using the authors’ replication code. All numbers here are simulated for demonstration; the firm’s real results appear in the paper.

---

## Overview
A compact, reproducible workflow for turning an A/B test (RCT) into a targeted policy with Causal ML:
1. Overall Intent-to-Treat (ATE)
2. Heterogeneous effects (CATE) via causal forests
3. “Who to treat” policy rule
4. Off-policy evaluation (IPS)  
Plus SHAP-based interpretability.

**Original replication code** is under **`src/`**. The generalized, article-oriented code is in the repository root.

---

## Repository Structure

```bash
├─ src/                         # Authors’ original replication code
│  ├─ simulate_data.py
│  ├─ cml_training.py
│  └─ cml_evaluation.py         # Same evaluation script as in the main branch
├─ data/                        # Train/test CSVs
├─ model/                       # Saved model
├─ plots/                       # Generated figures
├─ simulate_data_generalized.ipynb
├─ overall_treatment_effects.ipynb
├─ cml_training_generalized.ipynb
├─ cml_evaluation.py            # Evaluation script (GATE & IPS)
├─ shap_surrogate_explain.ipynb
└─ requirements.txt
```



---

## Running Order
- **`simulate_data_generalized.ipynb`** → Based on `src/simulate_data.py`; creates synthetic train/test and notes how to generalize to other A/B tests.  
- **`overall_treatment_effects.ipynb`** → Estimates overall (train+test) ATE, standard error, and 95% CI.  
- **`cml_training_generalized.ipynb`** → Based on `src/cml_training.py`; tunes/trains a Causal Forest with guidance on hyperparameters and extensions.  
- **`cml_evaluation.py`** → Produces GATE and IPS plots on test data (same as `src/cml_evaluation.py`).  
- **`shap_surrogate_explain.ipynb`** → Explains which features drive predicted CATEs using a GradientBoostingRegressor surrogate + SHAP.

---

## Installation
```bash
python -m venv .venv
# Windows: 
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```


---

## Quick Start

1. **Data**: Provide `data/train_data.csv` and `data/test_data.csv` (schema: `T`, `Y`, and pre-treatment `X*` features).
2. **Overall ATE**: Run `overall_treatment_effects.ipynb` to compute ATE, SE, and 95% CI and to generate a Figure-2–style mean plot.
3. **Train CATE model**: Run `cml_training_generalized.ipynb` (saves `model/causal_forest_dml_model.pkl`).
4. **Evaluate policy**: Run `python cml_evaluation.py` → outputs `plots/gate_plot.pdf` and `plots/ips_plot.pdf`.
5. **Interpretability**: Run `shap_surrogate_explain.ipynb` for SHAP bar/beeswarm plots explaining predicted CATEs.

---

## License

MIT License. See [`LICENSE`](./LICENSE) for details.

---

## Reference

von Zahn, M., et al. *Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning.* **Marketing Science** (Jul–Aug 2025).

