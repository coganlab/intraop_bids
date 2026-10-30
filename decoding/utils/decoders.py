import json
import h5py
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import get_scorer


class NaNImputer3D(BaseEstimator, TransformerMixin):
    """Impute NaN values in 3D epoch arrays using per-channel training means.

    During ``fit``, computes the mean waveform for each channel across
    non-NaN training trials.  During ``transform``, replaces NaNs with the
    fitted channel mean.  Channel/time positions that are NaN for every
    training trial are filled with the mean value of good channels.

    Parameters
    ----------
    noise_scale : float, optional
        If > 0, adds Gaussian noise scaled to ``noise_scale * channel_std``
        to imputed values.  Useful for preventing identical imputed vectors.
        Defaults to 0 (deterministic mean imputation).
    strategy : str, optional
        The strategy to use for imputation.  If 'split', mean statistics are
        computed separately for train and test sets. If 'train', mean
        statistics are computed on the training set and applied to the test
        set. Defaults to 'split'.
    """

    def __init__(self, noise_scale=0.0, strategy='split'):
        self.noise_scale = noise_scale
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'train':
            self.fill_means_, self.channel_stds_ = self._get_fill_values(X)

        return self


    def transform(self, X, y=None):
        X_out = X.copy()
        nan_mask = np.isnan(X_out)

        if not np.any(nan_mask):
            return X_out

        if self.strategy == 'split':
            fill_means, channel_stds = self._get_fill_values(X)
        else:
            fill_means = self.fill_means_
            channel_stds = self.channel_stds_

        # Broadcast mean waveform and channel standard deviations
        means = np.broadcast_to(fill_means[None, :, :], X_out.shape)
        stds = np.broadcast_to(channel_stds[None, None, :], X_out.shape)

        # Mean imputation
        X_out[nan_mask] = means[nan_mask]

        # Add noise only at imputed positions
        if self.noise_scale > 0:
            noise = np.random.normal(
                loc=0,
                scale=self.noise_scale * stds[nan_mask],
            )
            X_out[nan_mask] += noise

        return X_out

    def _get_fill_values(self, X):
        """Compute mean waveforms and channel standard deviations.

        Parameters
        ----------
        X : ndarray, shape (trials, time, channels)

        Returns
        -------
        fill_means : ndarray, shape (time, channels)
            Mean waveform for each channel, with fallbacks for entirely
            missing channels.
        channel_stds : ndarray, shape (channels,)
            Standard deviation of trial-to-trial variability for each channel.
        """
        # Mean waveform across trials: (time, channels)
        channel_means = np.nanmean(X, axis=0)

        # Time-varying mean across available channels: (time,)
        time_mean = np.nanmean(channel_means, axis=1)

        # Global fallback if all channels are missing at a time point
        global_mean = np.nanmean(X)
        if np.isnan(global_mean):
            global_mean = 0.0

        time_mean = np.where(np.isnan(time_mean), global_mean, time_mean)

        # Fill entirely missing channels using the time-varying mean
        fill_means = np.where(
            np.isnan(channel_means),
            time_mean[:, None],
            channel_means,
        )

        # Estimate variability around each channel's mean waveform.
        # This prevents time-varying evoked responses from inflating the
        # noise estimate.
        residuals = X - channel_means[None, :, :]
        channel_stds = np.nanstd(residuals, axis=(0, 1))

        # Fallback for entirely missing channels
        valid_stds = channel_stds[np.isfinite(channel_stds)]
        fallback_std = np.mean(valid_stds) if valid_stds.size else 0.0

        channel_stds = np.where(
            np.isfinite(channel_stds),
            channel_stds,
            fallback_std,
        )

        return fill_means, channel_stds


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

        try:
            list(cv.split(X, y))
        except ValueError:
            cv = KFold(n_splits=self.nFolds, shuffle=True)

        for i in tqdm(range(self.nIter), desc='Decoding iterations'):
            score, y_pred = self.decode_iter(X, y, cv,
                                             compute_chance=compute_chance)
            print(f"Iteration: {i+1} - {self.scorer.__name__}: {score:.4f}",
                  flush=True)
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
            'scores': (self.scores.tolist()
                       if self.scores is not None else None),
            'y_preds': (self.y_preds.tolist()
                        if self.y_preds is not None else None),
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
        return f"<{obj.__class__.__module__}.{obj.__class__.__name__}>"
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
