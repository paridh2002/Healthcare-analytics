import json,sys,time,warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("pipeline.py").read().split("# ---------- models ----------")[0].replace("path = sys.argv[1]","path = sys.argv[1]"))
S=json.load(open("summary.json"))
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats
rfp=S["rf_params"]; xp=S["xgb_params"]
rf=lambda: RandomForestClassifier(class_weight="balanced",n_jobs=-1,random_state=SEED,**rfp)
xg=lambda: XGBClassifier(objective="multi:softprob",eval_metric="mlogloss",n_jobs=4,random_state=SEED,verbosity=0,**xp)
rs=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=SEED)
a=[];b=[];t0=time.time();tr=[];tx=[]
for tr_i,te_i in rs.split(X_full,y):
    Xa,Xb,ya,yb=X_full.iloc[tr_i],X_full.iloc[te_i],y[tr_i],y[te_i]
    t=time.time();m=rf().fit(Xa,ya);tr.append(time.time()-t);a.append(f1_score(yb,m.predict(Xb),average="macro"))
    t=time.time();m=xg().fit(Xa,ya,sample_weight=compute_sample_weight("balanced",ya));tx.append(time.time()-t);b.append(f1_score(yb,m.predict(Xb),average="macro"))
a=np.array(a);b=np.array(b);tt=stats.ttest_rel(b,a);w=stats.wilcoxon(b,a)
out={"rf_f1":a.mean(),"rf_sd":a.std(),"xgb_f1":b.mean(),"xgb_sd":b.std(),"t":tt.statistic,"p_t":tt.pvalue,"p_w":w.pvalue,"rf_fit_s":np.mean(tr),"xgb_fit_s":np.mean(tx)}
# learning curve
Xtr,Xte,ytr,yte=train_test_split(X_full,y,test_size=.2,stratify=y,random_state=SEED)
lc=[]
for f in [.2,.4,.6,.8,1.0]:
    r=[]
    for sd in range(3):
        idx=np.random.RandomState(sd).choice(len(Xtr),int(f*len(Xtr)),replace=False)
        ys=ytr[idx]
        m=xg().fit(Xtr.iloc[idx],ys,sample_weight=compute_sample_weight("balanced",ys)); r.append(f1_score(yte,m.predict(Xte),average="macro"))
    lc.append((f,np.mean(r)))
out["lc"]=lc; print(out); json.dump(out,open("extra.json","w"),default=float)
fig,ax=plt.subplots(figsize=(3.4,2.2)); ax.plot([f*100 for f,_ in lc],[v for _,v in lc],"o-"); ax.set_xlabel("Training data used (%)"); ax.set_ylabel("Test macro-F1"); ax.set_title("XGBoost learning curve"); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("figs/fig_lc.png",dpi=300)
