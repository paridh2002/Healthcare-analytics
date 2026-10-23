# Ensemble ML on CMS Provider Data (IRF General Information, Sep 2026)

Random Forest and XGBoost (plus baselines) classifying Inpatient Rehabilitation Facility
ownership profile (For-profit / Non-profit / Government) from CMS provider attributes.

NOTE: dataset 7t8x-u3ir contains no star rating, patient-experience or readmission fields,
so the target is ownership. To predict overall star ratings, join Hospital General Information,
HCAHPS and Unplanned Hospital Visits tables on the CMS Certification Number and change the target.

## Run
    pip install -r requirements.txt
    # place the CMS CSV in data/
    python pipeline.py data/Inpatient_Rehabilitation_Facility-General_Information_Sep2026.csv   # CV, tuning, test, figures
    python extra.py data/Inpatient_Rehabilitation_Facility-General_Information_Sep2026.csv      # significance test + learning curve

Outputs: results/*.csv|json and figs/*.png. CPU only; seed 42.
