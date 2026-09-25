"""Single-load Titanic EDA and leakage-safe modeling pipeline."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE = Path(__file__).resolve().parent
OUT = BASE / "artifacts"; OUT.mkdir(exist_ok=True)
CSV = BASE / "titanic.csv"; MODEL = BASE / "best_survival_pipeline.joblib"
NUM = ["pclass", "age", "sibsp", "parch", "fare"]
CAT = ["sex", "embarked"]

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=160); plt.close(fig)

def metrics(name, pipe, x_test, y_test):
    pred = pipe.predict(x_test); prob = pipe.predict_proba(x_test)[:, 1]
    return {"model": name, "accuracy": accuracy_score(y_test,pred),
            "precision": precision_score(y_test,pred,zero_division=0),
            "recall": recall_score(y_test,pred,zero_division=0),
            "f1": f1_score(y_test,pred,zero_division=0), "auc": roc_auc_score(y_test,prob),
            "confusion_matrix": confusion_matrix(y_test,pred).tolist(), "fpr": roc_curve(y_test,prob)[0], "tpr": roc_curve(y_test,prob)[1]}

def preprocessor(numeric=NUM, categorical=CAT):
    return ColumnTransformer([("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)])

def main():
    # This is the module's only sns.load_dataset call. The raw fallback is committed.
    df = sns.load_dataset("titanic")
    df.to_csv(CSV, index=False)
    missing = (df.isna().mean().mul(100)).loc[lambda s:s.gt(0)].round(3)
    print(df.info()); print(df.describe(include="all")); print("shape:",df.shape); print(missing)
    # <5%: drop embarked rows; 5-30%: median-impute age for EDA; deck (77%): drop;
    # embark_town duplicates embarked and is dropped.
    eda = df.drop(columns=["deck","embark_town"]).dropna(subset=["embarked"]).copy()
    eda["age"] = eda["age"].fillna(eda["age"].median())
    outliers={}
    for col in ["age","fare"]:
        q1,q3=eda[col].quantile([.25,.75]); outliers[col]=int(((eda[col]<q1-1.5*(q3-q1))|(eda[col]>q3+1.5*(q3-q1))).sum())
        fig,ax=plt.subplots(); sns.histplot(eda[col],kde=True,ax=ax); ax.set_title(f"{col.title()} histogram"); save(fig,f"{col}_histogram.png")
        fig,ax=plt.subplots(); sns.boxplot(x=eda[col],ax=ax); ax.set_title(f"{col.title()} box plot"); save(fig,f"{col}_boxplot.png")
    fare_stats={"mean":eda.fare.mean(),"median":eda.fare.median(),"mode":eda.fare.mode().iloc[0]}
    sex_class_rates = eda.groupby(["sex","pclass"]).survived.mean()
    rates={"sex":eda.groupby("sex").survived.mean().to_dict(),"pclass":eda.groupby("pclass").survived.mean().to_dict(),"sex_pclass":{f"{sex}_class_{pclass}": rate for (sex,pclass),rate in sex_class_rates.items()}}
    corr_cols=["survived","pclass","age","sibsp","parch","fare"]; corr=eda[corr_cols].corr()
    fig,ax=plt.subplots(figsize=(7,5)); sns.heatmap(corr,annot=True,cmap="vlag",ax=ax); save(fig,"correlation_heatmap.png")
    pairs=corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().abs().sort_values(ascending=False).head(2).to_dict()
    for name,plot in [("survival_by_sex.png",lambda: sns.barplot(data=eda,x="sex",y="survived")),("survival_by_class.png",lambda: sns.barplot(data=eda,x="pclass",y="survived")),("survival_by_sex_class.png",lambda: sns.barplot(data=eda,x="pclass",y="survived",hue="sex")),("fare_by_survival.png",lambda: sns.boxplot(data=eda,x="survived",y="fare"))]:
        fig,ax=plt.subplots(); plot(); save(fig,name)
    z=eda[["age","fare"]].apply(lambda s:(s-s.mean())/s.std(ddof=0)); z_summary=z.agg(["mean","std"]).to_dict()
    # Modeling rows use the original one-load dataframe; only low-missing embarked rows are dropped.
    model_df=df.dropna(subset=["embarked"])[NUM+CAT+["survived"]].copy(); X=model_df[NUM+CAT]; y=model_df.survived
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={"Logistic Regression":LogisticRegression(max_iter=2000),"Decision Tree":DecisionTreeClassifier(random_state=42,max_depth=5),"Random Forest":RandomForestClassifier(random_state=42,n_estimators=300)}
    fitted={}; rows=[]
    for name,est in models.items():
        pipe=Pipeline([("preprocessor",preprocessor()),("model",est)]); pipe.fit(Xtr,ytr); fitted[name]=pipe; m=metrics(name,pipe,Xte,yte); rows.append(m)
        fig,ax=plt.subplots(); sns.heatmap(m["confusion_matrix"],annot=True,fmt="d",cbar=False,ax=ax); ax.set_title(name); save(fig,f"{name.lower().replace(' ','_')}_confusion.png")
    fig,ax=plt.subplots()
    for m in rows: ax.plot(m["fpr"],m["tpr"],label=f"{m['model']} AUC={m['auc']:.3f}")
    ax.plot([0,1],[0,1],"--"); ax.legend(); ax.set_title("ROC curves"); save(fig,"roc_curves.png")
    tree=fitted["Decision Tree"]; names=tree.named_steps["preprocessor"].get_feature_names_out(); fig,ax=plt.subplots(figsize=(20,10)); plot_tree(tree.named_steps["model"],feature_names=names,class_names=["not survived","survived"],filled=True,ax=ax); save(fig,"decision_tree.png")
    imbalance=[]
    for label,pipe in [("baseline",Pipeline([("preprocessor",preprocessor()),("model",LogisticRegression(max_iter=2000))])),("class_weight_balanced",Pipeline([("preprocessor",preprocessor()),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))])),("smote",ImbPipeline([("preprocessor",preprocessor()),("smote",SMOTE(random_state=42)),("model",LogisticRegression(max_iter=2000))]))]:
        pipe.fit(Xtr,ytr); p=pipe.predict(Xte); imbalance.append({"strategy":label,"precision":precision_score(yte,p),"recall":recall_score(yte,p),"f1":f1_score(yte,p)})
    grid=GridSearchCV(Pipeline([("preprocessor",preprocessor()),("model",RandomForestClassifier(random_state=42,oob_score=True))]),{"model__n_estimators":[100,300],"model__max_depth":[None,8],"model__max_features":["sqrt",None]},cv=5,scoring="f1",n_jobs=-1); grid.fit(Xtr,ytr); best=grid.best_estimator_; oob=best.named_steps["model"].oob_score_; joblib.dump(best,MODEL); reload_ok=np.array_equal(best.predict(Xte),joblib.load(MODEL).predict(Xte))
    reg_x=model_df.drop(columns=["fare","survived"]); reg_y=model_df.fare; rxtr,rxte,rytr,ryte=train_test_split(reg_x,reg_y,test_size=.2,random_state=42)
    reg=Pipeline([("preprocessor",preprocessor(["pclass","age","sibsp","parch"],CAT)),("model",LinearRegression())]); reg.fit(rxtr,rytr); rp=reg.predict(rxte); r2=r2_score(ryte,rp); n=len(ryte); k=reg.named_steps["preprocessor"].transform(rxte).shape[1]; regression={"mae":mean_absolute_error(ryte,rp),"rmse":mean_squared_error(ryte,rp)**.5,"r2":r2,"adjusted_r2":1-(1-r2)*(n-1)/(n-k-1)}
    fig,ax=plt.subplots(); ax.scatter(rp,ryte-rp); ax.axhline(0,color="red"); ax.set_title("Fare regression residuals"); ax.set_xlabel("Predicted fare"); ax.set_ylabel("Residual"); save(fig,"regression_residuals.png")
    comparison=pd.DataFrame([{k:v for k,v in r.items() if k not in {"confusion_matrix","fpr","tpr"}} for r in rows]); comparison.to_csv(OUT/"classification_comparison.csv",index=False); pd.DataFrame(imbalance).to_csv(OUT/"imbalance_comparison.csv",index=False)
    report={"shape":df.shape,"missing_percent":missing.to_dict(),"outliers":outliers,"fare":fare_stats,"survival_rates":rates,"strongest_correlations":{str(k):v for k,v in pairs.items()},"zscore":z_summary,"class_balance":y.value_counts(normalize=True).to_dict(),"classification":comparison.to_dict(orient="records"),"imbalance":imbalance,"grid_best":grid.best_params_,"oob_score":oob,"regression":regression,"reload_predictions_match":bool(reload_ok)}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=float)); print(json.dumps(report,indent=2,default=float))
if __name__=="__main__": main()
