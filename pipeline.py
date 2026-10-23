"""Ensemble ML pipeline on CMS Provider Data (IRF General Information, Sep 2026).
Task: 3-class ownership-profile classification (For-profit / Non-profit / Government).
Run: python pipeline.py <csv_path>
"""
import sys, re, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
SEED = 42
path = sys.argv[1]
df = pd.read_csv(path, dtype=str)

# ---------- target ----------
tmap = {"For profit": "For-profit", "Physician": "For-profit",
        "Non-profit": "Non-profit", "Government": "Government"}
df["y"] = df["Ownership Type"].map(tmap)
classes = ["Government", "For-profit", "Non-profit"]
y = df["y"].map({c: i for i, c in enumerate(classes)}).values

# ---------- feature engineering ----------
d = pd.DataFrame(index=df.index)
cert = pd.to_datetime(df["Certification Date"], errors="coerce")
d["cert_year"] = cert.dt.year
d["facility_age"] = 2026 - cert.dt.year
d["cert_month"] = cert.dt.month
d["cms_region"] = df["CMS Region"].astype(int)
d["zip3"] = df["ZIP Code"].str[:3].astype(int)
d["zip_first"] = df["ZIP Code"].str[0].astype(int)
d["has_addr2"] = (df["Address Line 2"].str.strip() != "-").astype(int)
# structural: state one-hot
state = pd.get_dummies(df["State"], prefix="st").astype(int)
# name-derived
nm = df["Provider Name"].str.upper()
d["name_len"] = nm.str.len()
d["name_words"] = nm.str.split().str.len()
d["is_unit"] = nm.str.contains(r"\bUNIT\b|DISTINCT|REHAB UNIT|REHABILITATION UNIT").astype(int)
d["has_hospital"] = nm.str.contains("HOSPITAL").astype(int)
toks = nm.str.findall(r"[A-Z]+").explode()
top = [t for t, _ in toks.value_counts().items() if len(t) > 2 and t not in {"THE", "AND", "OF", "AT", "INC", "LLC"}][:40]
tokdf = pd.DataFrame({f"tok_{t}": nm.str.contains(rf"\b{t}\b").astype(int) for t in top})
county_freq = df["County/Parish"].map(df["County/Parish"].value_counts())
d["county_freq"] = county_freq.values
d["state_count"] = df["State"].map(df["State"].value_counts()).values

X_struct = pd.concat([d.drop(columns=["name_len", "name_words", "is_unit", "has_hospital"]), state], axis=1)
X_full = pd.concat([d, state, tokdf], axis=1)
print("features: structural", X_struct.shape[1], " full", X_full.shape[1], " n =", len(df))

# ---------- models ----------
cw = "balanced"
def models():
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw, C=1.0)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight=cw, random_state=SEED),
        "MLP (64-32)": make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, max_iter=600, early_stopping=True, random_state=SEED)),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, class_weight=cw, n_jobs=-1, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=500, class_weight=cw, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
                                 colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                                 n_jobs=4, random_state=SEED, verbosity=0),
    }
scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy", "f1m": "f1_macro",
           "mcc": __import__("sklearn.metrics", fromlist=["make_scorer"]).make_scorer(matthews_corrcoef)}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

def cv_all(X, label):
    rows = []
    for name, m in models().items():
        fit_params = {}
        if name == "XGBoost":
            fit_params = {"sample_weight": compute_sample_weight("balanced", y)}
        r = cross_validate(m, X, y, cv=skf, scoring=scoring, params=fit_params, n_jobs=1)
        rows.append({"model": name, "feature_set": label,
                     **{k: (r[f"test_{k}"].mean(), r[f"test_{k}"].std()) for k in scoring}})
        print(label, name, round(r["test_acc"].mean(), 4), round(r["test_f1m"].mean(), 4))
    return rows

res = cv_all(X_full, "full") + cv_all(X_struct, "structural")
flat = []
for r in res:
    flat.append({"model": r["model"], "feature_set": r["feature_set"],
                 **{f"{k}_mean": r[k][0] for k in scoring}, **{f"{k}_std": r[k][1] for k in scoring}})
cvdf = pd.DataFrame(flat); cvdf.to_csv("cv_results.csv", index=False)

# ---------- tuning + held-out test (full features) ----------
Xtr, Xte, ytr, yte = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=SEED)
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
rf_space = {"n_estimators": [200, 400, 600], "max_depth": [None, 8, 12, 20], "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.3]}
xgb_space = {"n_estimators": [150, 300, 500], "max_depth": [3, 4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
             "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
             "reg_lambda": [0.5, 1, 5]}
rf_s = RandomizedSearchCV(RandomForestClassifier(class_weight=cw, n_jobs=-1, random_state=SEED), rf_space,
                          n_iter=20, cv=inner, scoring="f1_macro", random_state=SEED, n_jobs=1).fit(Xtr, ytr)
sw = compute_sample_weight("balanced", ytr)
xgb_s = RandomizedSearchCV(XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_jobs=4,
                                         random_state=SEED, verbosity=0), xgb_space,
                           n_iter=20, cv=inner, scoring="f1_macro", random_state=SEED, n_jobs=1).fit(Xtr, ytr, sample_weight=sw)
print("RF best", rf_s.best_params_, rf_s.best_score_); print("XGB best", xgb_s.best_params_, xgb_s.best_score_)
tuned = {"Random Forest (tuned)": rf_s.best_estimator_, "XGBoost (tuned)": xgb_s.best_estimator_}
test_rows, preds = [], {}
for name, m in tuned.items():
    p = m.predict(Xte); preds[name] = p
    test_rows.append({"model": name, "acc": accuracy_score(yte, p), "bal_acc": balanced_accuracy_score(yte, p),
                      "f1_macro": f1_score(yte, p, average="macro"), "mcc": matthews_corrcoef(yte, p)})
# soft-voting ensemble of tuned RF + XGB
pv = (rf_s.best_estimator_.predict_proba(Xte) + xgb_s.best_estimator_.predict_proba(Xte)) / 2
p = pv.argmax(1); preds["Soft-vote RF+XGB"] = p
test_rows.append({"model": "Soft-vote RF+XGB", "acc": accuracy_score(yte, p), "bal_acc": balanced_accuracy_score(yte, p),
                  "f1_macro": f1_score(yte, p, average="macro"), "mcc": matthews_corrcoef(yte, p)})
testdf = pd.DataFrame(test_rows); testdf.to_csv("test_results.csv", index=False); print(testdf)
from sklearn.metrics import classification_report
rep = classification_report(yte, preds["XGBoost (tuned)"], target_names=classes, output_dict=True)
json.dump({"rf_params": {k: (None if v is None else (v if isinstance(v,(str,float)) else int(v))) for k, v in rf_s.best_params_.items()},
           "xgb_params": {k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in xgb_s.best_params_.items()},
           "rf_cv": rf_s.best_score_, "xgb_cv": xgb_s.best_score_, "report_xgb": rep,
           "n": len(df), "n_train": len(Xtr), "n_test": len(Xte),
           "class_counts": df["y"].value_counts().to_dict(), "n_feat_full": X_full.shape[1], "n_feat_struct": X_struct.shape[1]},
          open("summary.json", "w"), indent=1, default=str)

# ---------- importance ----------
fi_rf = pd.Series(rf_s.best_estimator_.feature_importances_, index=X_full.columns)
fi_xgb = pd.Series(xgb_s.best_estimator_.feature_importances_, index=X_full.columns)
pi = permutation_importance(xgb_s.best_estimator_, Xte, yte, n_repeats=20, scoring="f1_macro", random_state=SEED, n_jobs=1)
fi_perm = pd.Series(pi.importances_mean, index=X_full.columns)
pd.DataFrame({"rf_gini": fi_rf, "xgb_gain": fi_xgb, "xgb_perm": fi_perm}).sort_values("xgb_gain", ascending=False).to_csv("importances.csv")

# ---------- figures ----------
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
# 1 class distribution
fig, ax = plt.subplots(figsize=(3.4, 2.1))
vc = df["Ownership Type"].value_counts(); ax.bar(vc.index, vc.values, color="#4c72b0")
for i, v in enumerate(vc.values): ax.text(i, v + 8, str(v), ha="center", fontsize=7)
ax.set_ylabel("Facilities"); ax.set_title("Ownership type distribution"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("figs/fig_class.png", dpi=300); plt.close()
# 2 cert year by class
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for c, col in zip(classes, ["#55a868", "#c44e52", "#4c72b0"]):
    ax.hist(d.loc[df["y"] == c, "cert_year"].dropna(), bins=range(1960, 2030, 5), alpha=.6, label=c, color=col)
ax.set_xlabel("Certification year"); ax.set_ylabel("Facilities"); ax.legend(fontsize=6); ax.set_title("Certification year by class"); plt.tight_layout(); plt.savefig("figs/fig_year.png", dpi=300); plt.close()
# 3 model comparison
cmp = cvdf[cvdf.feature_set == "full"].set_index("model")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
order = cmp["f1m_mean"].sort_values().index
ax.barh(order, cmp.loc[order, "f1m_mean"], xerr=cmp.loc[order, "f1m_std"], color="#4c72b0", capsize=2)
ax.set_xlabel("Macro-F1 (5-fold CV)"); ax.set_title("Model comparison"); plt.tight_layout(); plt.savefig("figs/fig_models.png", dpi=300); plt.close()
# 4 ablation
fig, ax = plt.subplots(figsize=(3.4, 2.2))
pv_ = cvdf.pivot(index="model", columns="feature_set", values="f1m_mean").loc[["Logistic Regression", "Decision Tree", "MLP (64-32)", "Extra Trees", "Random Forest", "XGBoost"]]
pv_.plot.bar(ax=ax, color=["#4c72b0", "#dd8452"], width=.75); ax.set_ylabel("Macro-F1"); ax.set_title("Feature-set ablation"); ax.legend(fontsize=6); plt.xticks(rotation=30, ha="right", fontsize=6); plt.tight_layout(); plt.savefig("figs/fig_ablation.png", dpi=300); plt.close()
# 5 confusion matrix
fig, ax = plt.subplots(figsize=(3.0, 2.6))
ConfusionMatrixDisplay(confusion_matrix(yte, preds["XGBoost (tuned)"]), display_labels=classes).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost (tuned), test set"); plt.xticks(rotation=20, fontsize=6); plt.tight_layout(); plt.savefig("figs/fig_cm.png", dpi=300); plt.close()
# 6 importance
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, s, t in [(axs[0], fi_xgb, "XGBoost gain"), (axs[1], fi_perm, "XGBoost permutation (macro-F1 drop)")]:
    s.sort_values().tail(12).plot.barh(ax=ax, color="#4c72b0"); ax.set_title(t); ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig("figs/fig_imp.png", dpi=300); plt.close()
# 7 architecture
fig, ax = plt.subplots(figsize=(7.0, 1.7)); ax.axis("off")
steps = ["CMS IRF CSV\n(1,222 x 12)", "Cleaning &\nparsing dates", "Feature\nengineering\n(7+state+name)", "Stratified\n80/20 split", "RandomSearch\n5-fold tuning", "RF | XGBoost\n(+soft vote)", "Metrics &\nimportance"]
for i, s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 1.0 + .02, .2), .86, .6, fc="#dbe7f5", ec="#2b4c7e")); ax.text(i + .45, .5, s, ha="center", va="center", fontsize=6.5)
    if i < len(steps) - 1: ax.annotate("", xy=(i + 1.02, .5), xytext=(i + .88, .5), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(0, 7); ax.set_ylim(0, 1); plt.tight_layout(); plt.savefig("figs/fig_arch.png", dpi=300); plt.close()
print("done")
