import json
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import get_scorer


class DimRedReshape(BaseEstimator):

    def __init__(self, dim_red, n_components=None):
        self.dim_red = dim_red
        self.n_components = n_components

    def fit(self, X, y=None):
        X_r = X.reshape(X.shape[0], -1)
        self.transformer = self.dim_red(n_components=self.n_components)
        self.transformer.fit(X_r)
        return self

    def transform(self, X, y=None):
        X_r = X.reshape(X.shape[0], -1)
        X_dr = self.transformer.transform(X_r)
        return X_dr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
class decodeResultsBIDS:
    def __init__(self, model, nFolds, nIter, scores=None, y_preds=None,
                 scorer='balanced_accuracy'):
        self.model = model
        self.nFolds = nFolds
        self.nIter = nIter
        self.scores = scores
        self.y_preds = y_preds
        self.scorer = get_scorer(scorer)._score_func

    def run_decoding(self, X, y, compute_chance=False):
        scores = np.zeros(self.nIter)
        y_preds = np.zeros((self.nIter, len(y)))
        cv = StratifiedKFold(n_splits=self.nFolds, shuffle=True)

        # check that stratification is possible
        try:
            list(cv.split(X, y))
        except ValueError as e:
            cv = KFold(n_splits=self.nFolds, shuffle=True)

        for i in tqdm(range(self.nIter), desc='Decoding iterations'):
            score, y_pred = self.decode_iter(X, y, cv, compute_chance=compute_chance)
            print(f"Iteration: {i+1} - {self.scorer.__name__}: {score:.4f}", flush=True)
            scores[i] = score
            y_preds[i] = y_pred
        self.scores = scores
        self.y_preds = y_preds

    def decode_iter(self, X, y, cv, compute_chance=False):
        if compute_chance:
            y_pred = np.zeros(len(y))
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, _ = y[train_idx], y[test_idx]
                y_train = np.random.permutation(y_train)
                self.model.fit(X_train, y_train)
                fold_pred = self.model.predict(X_test)
                y_pred[test_idx] = fold_pred
        else:
            y_pred = cross_val_predict(self.model, X, y, cv=cv, n_jobs=-1)
        acc = self.scorer(y, y_pred)
        return acc, y_pred
    
    def to_dict(self):
        return {
            'model': make_json_serializable(self.model.get_params()),
            'nFolds': self.nFolds,
            'nIter': self.nIter,
            'scores': self.scores.tolist() if self.scores is not None else None,
            'y_preds': self.y_preds.tolist() if self.y_preds is not None else None,
            'scorer': self.scorer.__name__,
        }
    
    def save_results(self, bidsPath, overwrite=True):
        if self.scores is None or self.y_preds is None:
            raise ValueError("No results to save. Run decoding first.")
        with h5py.File(bidsPath, 'w' if overwrite else 'a') as f:
            f.create_dataset('scores', data=self.scores)
            f.create_dataset('y_preds', data=self.y_preds)
            f.attrs['model'] = json.dumps(make_json_serializable(
                self.model.get_params()))
            f.attrs['nFolds'] = self.nFolds
            f.attrs['nIter'] = self.nIter
            f.attrs['scorer'] = self.scorer.__name__
        print(f"Results saved to {bidsPath}")


def make_json_serializable(obj):
    """Convert sklearn parameter dict to JSON-safe representation."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif callable(obj):
        return f"<function {obj.__module__}.{obj.__name__}>"
    elif hasattr(obj, "__class__"):
        # Handle sklearn transformers, classes, etc.
        return f"<{obj.__class__.__module__}.{obj.__class__.__name__}>"
    else:
        try:
            json.dumps(obj)
            return obj  # already serializable
        except TypeError:
            return str(obj)
