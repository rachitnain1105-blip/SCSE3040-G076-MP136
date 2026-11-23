"""Three MLflow v1 runs; register the best run and set champion alias."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_processing import DATA_PATH, OUTPUT_PATH, TARGET, clean_data, load_data, save_clean_data
ROOT=Path(__file__).resolve().parents[1]
TRACKING_URI='sqlite:///mlflow.db'; EXPERIMENT='MP136_Parcel_Delivery_v1'; MODEL_NAME='mp136-parcel-delivery'

def make_features(df):
    X=df.drop(columns=[TARGET,'record_id','days_from_tracking_page'],errors='ignore').copy()
    dates=pd.to_datetime(X.pop('date'),errors='coerce')
    X['date_ordinal']=dates.map(lambda d:d.toordinal() if pd.notna(d) else np.nan)
    return X

def build_pipeline(estimator):
    numeric=['date_ordinal','distance_km','weight_kg','hub_backlog','pin_code_tier','declared_value_rs','transfers']
    categorical=['service','destination']
    prep=ColumnTransformer([
      ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    return Pipeline([('preprocess',prep),('model',estimator)])

def main():
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    params=yaml.safe_load((ROOT/'params.yaml').read_text())
    mlflow.set_tracking_uri(TRACKING_URI); mlflow.set_registry_uri(TRACKING_URI); mlflow.set_experiment(EXPERIMENT)
    data_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    clean=clean_data(load_data()); save_clean_data(clean,OUTPUT_PATH)
    X=make_features(clean); y=clean[TARGET].astype(float)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=float(params['validation_size']),random_state=int(params['random_state']))
    candidates=[
      ('random_forest_100',RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=1,random_state=42,n_jobs=-1)),
      ('random_forest_regularized',RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=-1)),
      ('extra_trees_200',ExtraTreesRegressor(n_estimators=200,max_depth=None,min_samples_leaf=2,random_state=42,n_jobs=-1))]
    results=[]
    for name,est in candidates:
        pipe=build_pipeline(est)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param('data_sha256',data_sha256); mlflow.log_param('candidate',name)
            mlflow.log_param('random_state',int(params['random_state'])); mlflow.log_param('validation_size',float(params['validation_size']))
            mlflow.log_param('training_rows',len(Xtr)); pipe.fit(Xtr,ytr)
            mae=float(mean_absolute_error(yv,pipe.predict(Xv))); mlflow.log_metric('mae_validation',mae)
            mlflow.sklearn.log_model(sk_model=pipe,artifact_path='model',serialization_format='cloudpickle')
            results.append({'candidate':name,'run_id':run.info.run_id,'mae_validation':mae,'model':pipe})
            print(f'{name}: MAE={mae:.5f}; run_id={run.info.run_id}')
    best=min(results,key=lambda x:x['mae_validation'])
    version=mlflow.register_model(f"runs:/{best['run_id']}/model",MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME,'champion',version.version)
    (ROOT/'models').mkdir(exist_ok=True); dump(best['model'],ROOT/'models/champion.joblib')
    summary={'experiment_name':EXPERIMENT,'registered_model_name':MODEL_NAME,'champion_version':str(version.version),
      'best_run_id':best['run_id'],'best_candidate':best['candidate'],'best_validation_mae':best['mae_validation'],
      'data_sha256':data_sha256,'candidate_runs':[{k:v for k,v in x.items() if k!='model'} for x in results]}
    (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'reports/stage2_training_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
