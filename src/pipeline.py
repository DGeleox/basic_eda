numeric = ["Age", "Fare", "SibSp", "Parch"]
categorical = ["Pclass", "Sex", "Embarked", "Cabin"]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def inference_titanic_pipeline(X_train, X_test, y_train, y_test):
    num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_tf, numeric),
        ("cat", cat_tf, categorical)
    ])
    
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC‑AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))