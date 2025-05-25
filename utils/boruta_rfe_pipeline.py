import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from joblib import Parallel, delayed
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback

class Tuner(BaseEstimator, TransformerMixin):
    """
    Generic tuner transformer using Optuna for classification/regression/multiclass.
    prefix distinguishes stage: 'pre_', 'mid_', 'final_'.
    """
    def __init__(self, estimator_name, problem_type, base_params, prefix,
                 tune_n_estimators, cv, random_state):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.base_params = base_params or {}
        self.prefix = prefix
        self.tune_n_estimators = tune_n_estimators
        self.cv = cv
        self.random_state = random_state

    def _xgb_objective(self, trial, X, y):
        # shared params
        p = {'verbosity':0, 'seed':self.random_state}
        # problem-specific
        if self.problem_type == 'binary':
            p.update({'objective':'binary:logistic','eval_metric':'logloss'})
        elif self.problem_type == 'multiclass':
            p.update({'objective':'multi:softprob','eval_metric':'mlogloss',
                      'num_class':len(np.unique(y))})
        else:
            p.update({'objective':'reg:squarederror','eval_metric':'rmse'})
        # hyperparameters
        p.update({
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        })
        n_est = trial.suggest_int('n_estimators', 100, 1000) if self.tune_n_estimators else self.base_params.get('n_estimators', 1000)
        Model = XGBClassifier if self.problem_type in ['binary','multiclass'] else XGBRegressor
        model = Model(**p, n_estimators=n_est, tree_method='hist', random_state=self.random_state)
        # cross-val
        scores = []
        for tr_idx, val_idx in self.cv.split(X):
            model.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                early_stopping_rounds=50,
                callbacks=[XGBoostPruningCallback(trial, 
                              "validation_0-mlogloss" if self.problem_type=='multiclass' 
                              else "validation_0-logloss" if self.problem_type=='binary'
                              else "validation_0-rmse")],
                verbose=False
            )
            pred = model.predict(X[val_idx])
            if self.problem_type in ['binary','multiclass']:
                scores.append(np.mean(pred==y[val_idx]))
            else:
                scores.append(-np.sqrt(((pred-y[val_idx])**2).mean()))
        return np.mean(scores)

    def _tune(self, X, y):
        params = self.base_params.copy()
        params.update(getattr(self, 'prev_params', {}))
        best_params = {}
        # LightGBM
        if self.estimator_name.lower() == 'lightgbm':
            if self.problem_type == 'binary': params.setdefault('objective','binary')
            elif self.problem_type == 'multiclass':
                params.setdefault('objective','multiclass')
                params['num_class']=len(np.unique(y))
            else: params.setdefault('objective','regression')
            train_set = lgb.Dataset(X, label=y)
            tuner = LightGBMTunerCV(params, train_set, folds=self.cv,
                                    early_stopping_rounds=50, verbose_eval=False,
                                    seed=self.random_state)
            tuner.run()
            best_params = tuner.best_params
        elif self.estimator_name.lower() == 'xgboost':
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(lambda t: self._xgb_objective(t, X, y), n_trials=50)
            best_params = study.best_params
            if not self.tune_n_estimators:
                best_params.setdefault('n_estimators', params.get('n_estimators',1000))
        else:
            best_params = params.copy()
        mlflow.log_params({f"{self.prefix}{k}": v for k,v in best_params.items()})
        return best_params

    def fit(self, X, y, **fit_params):
        self.prev_params = fit_params.get('prev_params', {})
        tuned = self._tune(X, y)
        self.params_ = self.prev_params.copy()
        self.params_.update(tuned)
        return self

    def transform(self, X): return X


class BorutaStep(BaseEstimator, TransformerMixin):
    """Boruta feature selection step with MLflow logging."""
    def __init__(self, estimator_name, problem_type, random_state, max_iter):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state
        self.max_iter = max_iter

    def _make_estimator(self, params):
        if self.estimator_name == 'xgboost':
            Model = XGBClassifier if self.problem_type in ['binary','multiclass'] else XGBRegressor
            return Model(**params, tree_method='hist', random_state=self.random_state)
        if self.estimator_name == 'lightgbm':
            Model = lgb.LGBMClassifier if self.problem_type in ['binary','multiclass'] else lgb.LGBMRegressor
            return Model(**params, random_state=self.random_state)
        Model = RandomForestClassifier if self.problem_type in ['binary','multiclass'] else RandomForestRegressor
        return Model(**params, n_jobs=-1, random_state=self.random_state)

    def fit(self, X, y, **fit_params):
        params = fit_params.get('prev_params', {})
        est = self._make_estimator(params)
        n_est = params.get('n_estimators','auto')
        boruta = BorutaPy(est, n_estimators=n_est,
                          max_iter=self.max_iter,
                          random_state=self.random_state)
        boruta.fit(X, y)
        self.support_ = boruta.support_
        mlflow.log_metric('boruta_n_features', int(self.support_.sum()))
        return self

    def transform(self, X): return X[:, self.support_]

class RFEPruner(BaseEstimator, TransformerMixin):
    """Redundancy pruning via blocked RFE with MLflow logging."""
    def __init__(self, estimator_name, problem_type, random_state, cv, step, survival_threshold):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state
        self.cv = cv
        self.step = step
        self.survival_threshold = survival_threshold

    def _make_estimator(self, params):
        if self.estimator_name == 'xgboost':
            Model = XGBClassifier if self.problem_type in ['binary','multiclass'] else XGBRegressor
            return Model(**params, tree_method='hist', random_state=self.random_state)
        if self.estimator_name == 'lightgbm':
            Model = lgb.LGBMClassifier if self.problem_type in ['binary','multiclass'] else lgb.LGBMRegressor
            return Model(**params, random_state=self.random_state)
        Model = RandomForestClassifier if self.problem_type in ['binary','multiclass'] else RandomForestRegressor
        return Model(**params, n_jobs=-1, random_state=self.random_state)

    def fit(self, X, y, **fit_params):
        params = fit_params.get('prev_params', {})
        est = self._make_estimator(params)
        supports=[]
        for tr_idx, val_idx in self.cv.split(X):
            sel = RFE(est, n_features_to_select=1, step=self.step)
            sel.fit(X[tr_idx], y[tr_idx])
            supports.append(sel.support_)
        surv = np.mean(supports,axis=0)
        self.support_ = surv >= self.survival_threshold
        mlflow.log_metric('stable_n_features', int(self.support_.sum()))
        return self

    def transform(self, X): return X[:, self.support_]

class FinalEstimator(BaseEstimator):
    """Final estimator step for classification/regression/multiclass."""
    def __init__(self, estimator_name, problem_type, random_state):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        params=fit_params.get('prev_params',{})
        if self.estimator_name=='xgboost':
            Model=XGBClassifier if self.problem_type in ['binary','multiclass'] else XGBRegressor
            self.model_=Model(**params,tree_method='hist',random_state=self.random_state)
        elif self.estimator_name=='lightgbm':
            Model=lgb.LGBMClassifier if self.problem_type in ['binary','multiclass'] else lgb.LGBMRegressor
            self.model_=Model(**params,random_state=self.random_state)
        else:
            Model=RandomForestClassifier if self.problem_type in ['binary','multiclass'] else RandomForestRegressor
            self.model_=Model(**params,n_jobs=-1,random_state=self.random_state)
        self.model_.fit(X,y)
        return self

    def predict(self, X): return self.model_.predict(X)


def build_pipeline(estimator_name, problem_type, base_params, pre_tune, mid_tune,
                   n_inner_splits, rfe_step, survival_threshold,
                   boruta_max_iter, random_state):
    cv=TimeSeriesSplit(n_splits=n_inner_splits)
    steps=[]
    steps.append(('pre_tune',Tuner(estimator_name,problem_type,base_params,'pre_',False,cv,random_state)))
    steps.append(('boruta',BorutaStep(estimator_name,problem_type,random_state,boruta_max_iter)))
    steps.append(('mid_tune',Tuner(estimator_name,problem_type,base_params,'mid_',True,cv,random_state)))
    steps.append(('rfe',RFEPruner(estimator_name,problem_type,random_state,cv,rfe_step,survival_threshold)))
    steps.append(('final_tune',Tuner(estimator_name,problem_type,base_params,'final_',True,cv,random_state)))
    steps.append(('clf',FinalEstimator(estimator_name,problem_type,random_state)))
    return Pipeline(steps)


def run_parallel(X, y, estimator_name='randomforest', problem_type='classification', base_params=None,
                 pre_tune=False, mid_tune=True, n_outer_splits=5,
                 forecast_horizon=50, n_inner_splits=3,
                 rfe_step=1, survival_threshold=0.7,
                 boruta_max_iter=50, random_state=42, n_jobs=-1,
                 metrics=None):
    """
    metrics: dict of metric_name->callable(y_true,y_pred)
    """
    mlflow.set_experiment('TimeSeriesFeatureSelection')
    tscv_outer=TimeSeriesSplit(n_splits=n_outer_splits,test_size=forecast_horizon)
    if metrics is None:
        if problem_type in ['binary','multiclass']:
            metrics={'accuracy':lambda y, p: np.mean(p==y)}
        else:
            metrics={'neg_rmse':lambda y,p: -np.sqrt(((p-y)**2).mean())}
    def proc(fold,tr,te):
        pipe=build_pipeline(estimator_name,problem_type,base_params,pre_tune,mid_tune,
                            n_inner_splits,rfe_step,survival_threshold,
                            boruta_max_iter,random_state)
        X_tr,y_tr=X[tr],y[tr]
        X_te,y_te=X[te],y[te]
        with mlflow.start_run(run_name=f'fold_{fold}'):
            pipe.fit(X_tr,y_tr)
            preds=pipe.predict(X_te)
            for name,fn in metrics.items():
                val=fn(y_te,preds)
                mlflow.log_metric(name,val)
            return {'fold':fold,**{name:fn(y_te,preds) for name,fn in metrics.items()}}
    results=Parallel(n_jobs=n_jobs)(delayed(proc)(i,tr,te)
                                   for i,(tr,te) in enumerate(tscv_outer.split(X)))
    return pd.DataFrame(results)

if __name__=='__main__':
    # Classification multi example
    X_cls=np.random.randn(1000,20);
    y_cls=np.random.randint(0,4,1000)

    metrics_cls={'accuracy':lambda y,p: np.mean(p==y),
                 'f1_macro':lambda y,p: None  # placeholder
                }
    df_cls=run_parallel(X_cls,y_cls,'xgboost','multiclass',base_params={},
                        pre_tune=True,mid_tune=True,n_outer_splits=3,
                        forecast_horizon=50,n_inner_splits=2,
                        rfe_step=1,survival_threshold=0.5,
                        boruta_max_iter=20,random_state=0,n_jobs=2,
                        metrics=metrics_cls)
    print("Multiclass results:\n",df_cls)

    # Multiclass RandomForest example
    df_rf = run_parallel(X_cls, y_cls, 'randomforest', 'multiclass', base_params={'n_estimators': 100},
                        pre_tune=False, mid_tune=False, n_outer_splits=3,
                        forecast_horizon=50, n_inner_splits=2,
                        rfe_step=1, survival_threshold=0.5,
                        boruta_max_iter=20, random_state=0, n_jobs=2,
                        metrics=metrics_cls
    )
    print("RandomForest multiclass results: ", df_rf)

    # Regression example
    X_reg=np.random.randn(1000,20); y_reg=np.random.randn(1000)
    df_reg=run_parallel(X_reg,y_reg,'lightgbm','regression',base_params={},
                        pre_tune=True,mid_tune=True,n_outer_splits=3,
                        forecast_horizon=50,n_inner_splits=2,
                        rfe_step=1,survival_threshold=0.5,
                        boruta_max_iter=20,random_state=0,n_jobs=2)
    print("Regression results:\n",df_reg)
