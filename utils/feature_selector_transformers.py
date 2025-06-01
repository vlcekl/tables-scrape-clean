from sklearn.base import BaseEstimator, TransformerMixin
from BorutaShap import BorutaShap

class BorutaSHAPStep(BaseEstimator, TransformerMixin):
    """
    All-relevant feature selection via Boruta-Shap.

    Parameters
    ----------
    model : str or estimator, default='lightgbm'
        Which base learner to explain with SHAP.  Accepts:
        - a string: 'lightgbm', 'xgboost', 'random_forest'
        - or any fitted tree-based estimator instance (must implement .fit/.predict)

    importance_measure : {'shap','gain','perm'}, default='shap'
        Which importance to use for testing.  'shap' is recommended for both
        RandomForest and GBMs; 'gain' will fall back to built-in tree gain;
        'perm' will use permutation importances.

    classification : bool, default=True
        Whether the problem is classification (True) or regression (False).

    percentile : int in [1,100], default=100
        When sampling data internally to approximate SHAP, what percentile of
        data to keep on each iteration (higher => more samples => slower but
        more precise).

    iterations : int, default=100
        Number of Boruta iterations to run (i.e. shadow-feature comparisons).

    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    support_ : 1d boolean array of length n_features
        Mask of features accepted as “important” by BorutaShap.

    feature_importances_ : pandas.Series
        Mean absolute SHAP (or chosen) importances for each original feature.
    """

    def __init__(
        self,
        model: Any = 'lightgbm',
        importance_measure: str = 'shap',
        classification: bool = True,
        percentile: int = 100,
        iterations: int = 100,
        random_state: Optional[int] = None
    ):
        self.model = model
        self.importance_measure = importance_measure
        self.classification = classification
        self.percentile = percentile
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the BorutaShap selector on X, y.
        """
        # initialize BorutaShap
        self._selector = BorutaShap(
            model=self.model,
            importance_measure=self.importance_measure,
            classification=self.classification,
            percentile=self.percentile,
            iterations=self.iterations,
            random_state=self.random_state
        )

        # BorutaShap expects numpy or pd.DataFrame
        self._selector.fit(X=X, y=y, normalize=True)

        # accepted_ is list of selected feature names
        accepted = set(self._selector.accepted)
        self.support_ = [col in accepted for col in X.columns]

        # store importances for diagnostics
        importances = dict(zip(X.columns, self._selector.shap_values_mean if hasattr(self._selector, 'shap_values_mean') else []))
        self.feature_importances_ = pd.Series(importances).loc[X.columns]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to only the accepted features.
        """
        return X.loc[:, self.support_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FBEDBackwardStep(BaseEstimator, TransformerMixin):
    """Backward elimination with inner random-train / fixed-year validation CV"""
    def __init__(self, estimator_name, problem_type, cv, random_state, metric_name, metrics):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.cv = cv
        self.random_state = random_state
        self.metric_name = metric_name
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        cols = list(X.columns)
        baseline = self._score_subset(X, y)
        support = cols.copy()
        for col in cols:
            subset = [c for c in support if c != col]
            score = self._score_subset(X[subset], y)
            if score >= baseline:
                support.remove(col)
                baseline = score
        self.support_ = [c in support for c in cols]
        return self

    def _score_subset(self, X_sub, y):
        model = make_estimator(self.estimator_name, self.problem_type, {}, self.random_state)
        scores = []
        for tr, val in self.cv.split(X_sub, y):
            model.fit(X_sub.iloc[tr], y.iloc[tr])
            pred = model.predict(X_sub.iloc[val])
            scores.append(self.metrics[self.metric_name][0](y.iloc[val], pred))
        return np.mean(scores)

    def transform(self, X: pd.DataFrame):
        return X.loc[:, X.columns[self.support_]]


class FinalEstimator(BaseEstimator):
    def __init__(self, estimator_name, problem_type, random_state):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        self.model_ = make_estimator(self.estimator_name, self.problem_type, {}, self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(X)


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        # ... any pre-processing steps ...
        ('borutashap', BorutaSHAPStep(
            model='lightgbm',
            importance_measure='shap',
            classification=(problem_type in ['binary','multiclass']),
            percentile=80,
            iterations=50,
            random_state=42
        )),
        # optional: tuner, backward-elim, or directly final estimator
        ('final', FinalEstimator(estimator_name, problem_type, random_state))
    ])
