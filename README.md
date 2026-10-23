# Ensemble Machine Learning for Profiling CMS Inpatient Rehabilitation Facilities

Random Forest, XGBoost and a soft-voting ensemble on CMS Provider Data
(Inpatient Rehabilitation Facility - General Information, September 2026, dataset `7t8x-u3ir`).

## Important note on scope
The IRF General Information file has 1,222 rows and 12 administrative columns. It does **not**
contain an overall star rating, patient-experience scores or readmission rates. The supervised
target is therefore **ownership profile** (For-profit / Non-profit / Government; the 8
physician-owned facilities are merged into For-profit), not hospital quality.
To predict overall star ratings, join Hospital General Information, HCAHPS and Unplanned Hospital
Visits tables on the CMS Certification Number (CCN) and replace the target in `pipeline.py`.

## Repository structure
```
.
├── pipeline.py        # feature engineering, 5-fold CV, tuning, test evaluation, figures
├── extra.py           # repeated-CV significance test (RF vs XGBoost) and learning curve
├── requirements.txt
├── data/              # CMS CSV (download from data.cms.gov/provider-data/dataset/7t8x-u3ir)
├── results/           # cv_results.csv, test_results.csv, importances.csv, summary.json, extra.json
└── figs/              # figures used in the report and slides
```

## Setup and run
```bash
pip install -r requirements.txt
python pipeline.py data/Inpatient_Rehabilitation_Facility-General_Information_Sep2026.csv
python extra.py    data/Inpatient_Rehabilitation_Facility-General_Information_Sep2026.csv
```
CPU only. Runs in a few minutes. Random seed 42. Tested with Python 3.12, scikit-learn 1.8, XGBoost 3.4.
Note: `pipeline.py` writes outputs to the current directory, so run it from the repository root
after creating the `figs/` folder (`mkdir -p figs`).

## Method summary
- **Features:** 61 structural (certification year/month/age, CMS region, ZIP prefixes, Address-2 flag,
  county frequency, state count, state one-hot) and 105 full (adds name length, word count,
  unit/hospital flags and 40 name-token indicators).
- **Models:** majority baseline, logistic regression, decision tree, extra trees, MLP (64-32),
  Random Forest, XGBoost, soft-vote RF+XGBoost.
- **Evaluation:** stratified 5-fold CV; 80/20 stratified split (977 train / 245 test);
  RandomizedSearchCV (20 configurations, macro-F1); accuracy, balanced accuracy, macro-F1, MCC;
  permutation importance; feature-set ablation; 3x5 repeated-CV paired t-test and Wilcoxon test.

## Key results (held-out test set, n = 245)
| Model | Accuracy | Balanced acc. | Macro-F1 | MCC |
|---|---|---|---|---|
| Random Forest (tuned) | 0.727 | 0.569 | 0.573 | 0.511 |
| XGBoost (tuned) | 0.714 | 0.670 | 0.629 | 0.535 |
| Soft-vote RF+XGB | 0.735 | 0.644 | 0.636 | 0.541 |

Majority-class baseline accuracy: 0.466. XGBoost vs RF (3x5 CV macro-F1): 0.595 vs 0.563, p = 0.003.
Top drivers: operator-name tokens (e.g. ENCOMPASS), UNIVERSITY, HOSPITAL, state and certification year.

## Limitations
Target is ownership, not quality. Name tokens partly encode operator brand. The government class has
only 100 samples (F1 about 0.37). Feature importance is associative, not causal.

## Data and citation
CMS Provider Data Catalog, "Inpatient Rehabilitation Facility - General Information",
https://data.cms.gov/provider-data/dataset/7t8x-u3ir

## License
Add the license of your choice (e.g., MIT) before publishing.
