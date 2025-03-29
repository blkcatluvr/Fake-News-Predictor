import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def trainModels(X_train,y_train):
    xgbModel = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    rfModel = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Tree depth
        random_state=42
        )
    xgbModel.fit(X_train, y_train)
    rfModel.fit(X_train, y_train)
        
    joblib.dump(xgbModel,"models/xgbModel.pkl")
    joblib.dump(rfModel, "models/rfModel.pkl")
    return xgbModel, rfModel