""" Cross-Patient Decoder Classes

Author: Zac Spalding
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


class crossPtDecoder(BaseEstimator):

    def preprocess_train(self, X, y=None):
        pass

    def preprocess_test(self, X, y=None):
        pass

    def filter_labels(self, X_p, y, y_p):
        keep_idx = self._get_shared_labels_idx(y, y_p)
        return X_p[keep_idx], y_p[keep_idx]
    
    def _get_shared_labels_idx(self, y_tar, y_pool):
        tar_labels = np.unique(y_tar)
        keep_idx = np.isin(y_pool, tar_labels)
        return keep_idx

    def fit(self, X, y, **kwargs):
        X_p, y_p = self.preprocess_train(X, y, **kwargs)
        X_p, y_p = self.filter_labels(X_p, y, y_p)
        return self.decoder.fit(X_p, y_p)

    def predict(self, X):
        X_p = self.preprocess_test(X)
        return self.decoder.predict(X_p)

    def score(self, X, y, **kwargs):
        X_p = self.preprocess_test(X)
        return self.decoder.score(X_p, y, **kwargs)


class crossPtDecoder_sepDimRed(crossPtDecoder):
    """ Cross-Patient Decoder with separate PCA for each patient. """

    def __init__(self, cross_pt_data, decoder, dim_red=PCA, n_comp=0.8,
                 tar_in_train=True):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, **kwargs):
        cross_pt_trials = [x.shape[0] for x, _, _ in self.cross_pt_data]
        # reshape features to be 2D (preserve last dimension for reduction)
        X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _, _ in
                     self.cross_pt_data]
        X_tar_r = X.reshape(-1, X.shape[-1])
        # reduce dimensionality of cross-patient data
        X_cross_dr = [self.dim_red(n_components=self.n_comp).fit_transform(x)
                      for x in X_cross_r]

        # reduce dimensionality of target data, saving dim. red. object for
        # test set
        tar_dr = self.dim_red(n_components=self.n_comp)
        X_tar_dr = tar_dr.fit_transform(X_tar_r)
        self.tar_dr = tar_dr

        # use the same latent dimensionality for all patients
        lat_dims = [X_tar_dr.shape[-1]] + [x.shape[-1] for x in X_cross_dr]
        self.common_dim = min(lat_dims)
        X_tar_dr = X_tar_dr[:, :self.common_dim]
        X_cross_dr = [x[:, :self.common_dim] for x in X_cross_dr]

        # reshape for concatenation
        X_cross_dr = [x.reshape(cross_pt_trials[i], -1, x.shape[-1]) for i, x
                      in enumerate(X_cross_dr)]
        X_cross_dr = [x.reshape(x.shape[0], -1) for x in X_cross_dr]
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1)

        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_cross_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_cross_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        X_r = X.reshape(-1, X.shape[-1])
        X_dr = self.tar_dr.transform(X_r)
        X_dr = X_dr[:, :self.common_dim]
        return X_dr.reshape(X.shape[0], -1)


class crossPtDecoder_sepAlign(crossPtDecoder):
    """ Cross-Patient Decoder with CCA alignment of separate dimensionality
    reductions for different patients."""

    def __init__(self, cross_pt_data, decoder, aligner, dim_red=PCA,
                 n_comp=0.8, tar_in_train=True):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp
        self.aligner = aligner
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        cross_pt_trials = [x.shape[0] for x, _, _ in self.cross_pt_data]
        # reshape features to be 2D (preserve last dimension for reduction)
        X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _, _ in
                     self.cross_pt_data]
        X_tar_r = X.reshape(-1, X.shape[-1])
        # reduce dimensionality of cross-patient data
        X_cross_dr = [self.dim_red(n_components=self.n_comp).fit_transform(x)
                      for x in X_cross_r]

        # reduce dimensionality of target data, saving dim. red. object for
        # test set
        tar_dr = self.dim_red(n_components=self.n_comp)
        X_tar_dr = tar_dr.fit_transform(X_tar_r)
        self.tar_dr = tar_dr

        # reshape back to 3D
        X_cross_dr = [x.reshape(cross_pt_trials[i], -1, x.shape[-1]) for i, x
                      in enumerate(X_cross_dr)]
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1, X_tar_dr.shape[-1])

        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # align data to target patient
        self.algns = [self.aligner() for _ in range(len(self.cross_pt_data))]
        X_algn_dr = []
        for i, algn in enumerate(self.algns):
            algn.fit(X_tar_dr, X_cross_dr[i], y_align, y_align_cross[i])
            X_algn_dr.append(algn.transform(X_cross_dr[i]))

        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

        # concatenate cross-patient data
        # X_pool = np.vstack([X_tar_dr] + X_algn_dr)
        # y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        X_r = X.reshape(-1, X.shape[-1])
        X_dr = self.tar_dr.transform(X_r)
        return X_dr.reshape(X.shape[0], -1)


class crossPtDecoder_jointDimRed(crossPtDecoder):
    """ Cross-Patient Decoder with joint dimensionality reduction to align and
    pool patients."""

    def __init__(self, cross_pt_data, decoder, joint_dr_method, n_comp=0.8,
                 tar_in_train=True):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.joint_dr_method = joint_dr_method
        self.n_comp = n_comp
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # extract features from cross pt data
        X_cross = [x for x, _, _ in self.cross_pt_data]

        # joint dimensionality reduction
        self.joint_dr = self.joint_dr_method(n_components=self.n_comp)
        X_joint_dr = self.joint_dr.fit_transform(
                                                [X] + X_cross,
                                                [y_align] + y_align_cross)
        X_tar_dr, X_algn_dr = X_joint_dr[0], X_joint_dr[1:]

        # reshape to trialx x features
        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

        # concatenate cross-patient data
        # X_pool = np.vstack([X_tar_dr] + X_algn_dr)
        # y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        X_dr = self.joint_dr.transform(X, idx=0)
        return X_dr.reshape(X.shape[0], -1)


class crossPtDecoder_mcca(crossPtDecoder):
    """ Cross-patient Decoder with MCCA to align and pool patients. """

    def __init__(self, cross_pt_data, decoder, aligner, n_comp=10, regs=0.5,
                 pca_var=1, tar_in_train=True):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.aligner = aligner
        self.n_comp = n_comp
        self.regs = regs
        self.pca_var = pca_var
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # extract features from cross pt data
        X_cross = [x for x, _, _ in self.cross_pt_data]

        # joint dimensionality reduction
        self.aligner = self.aligner(n_components=self.n_comp, regs=self.regs,
                                    pca_var=self.pca_var)
        X_mcca = self.aligner.fit_transform([X] + X_cross,
                                            [y_align] + y_align_cross)
        X_tar_dr, X_algn_dr = X_mcca[0], X_mcca[1:]

        # reshape to trialx x features
        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

        # concatenate cross-patient data
        # X_pool = np.vstack([X_tar_dr] + X_algn_dr)
        # y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        X_mcca = self.aligner.transform(X, idx=0)
        return X_mcca.reshape(X.shape[0], -1)


class crossPtDecoder_twSepAlign(crossPtDecoder):
    """Cross-Patient Decoder with CCA alignment and subsequent decoding.
    Allows for separate time windows for alignment and decoding."""

    def __init__(self, cross_pt_data, decoder, aligner, tw_full, tw_align,
                 tw_decode, dim_red=PCA, n_comp=0.8, tar_in_train=True):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp
        self.aligner = aligner
        self.tw_full = tw_full
        self.tw_align = tw_align
        self.tw_decode = tw_decode
        self.tar_in_train = tar_in_train
        self.y_align = None

        # cache containers
        self.cross_pt_cache = None  # dict patient_idx -> {"pca": obj, "X_align": arr, "y_align": arr}
        self.tar_dr = None          # PCA object for target patient (set per fit)

    # -------------------------------------------------------
    # One-time preprocessing of all cross-patient datasets
    # -------------------------------------------------------
    def build_cross_patient_cache(self):
        """Precompute PCA-reduced cross-patient features once."""
        if self.cross_pt_cache is not None:
            return self.cross_pt_cache  # already built

        cache = {}
        for i, (X, y, y_align) in enumerate(self.cross_pt_data):
            # time handling
            t_data = self.get_time_range(X.shape[1])
            align_idx = self.get_time_idx(t_data, self.tw_align)

            # flatten align window
            X_r = X[:, align_idx, :].reshape(-1, X.shape[-1])

            # fit PCA
            pca = self.dim_red(n_components=self.n_comp).fit(X_r)
            X_dr = pca.transform(X_r)
            X_dr = X_dr.reshape(X.shape[0], -1, X_dr.shape[-1])

            cache[i] = {
                "pca": pca,
                "X_align": X_dr,
                "y": y,
                "y_align": y_align,
            }
        self.cross_pt_cache = cache
        return cache

    # -------------------------------------------------------
    # Training-time preprocessing
    # -------------------------------------------------------
    def preprocess_train(self, X, y):
        cache = self.build_cross_patient_cache()
        cross_pt_trials = [x.shape[0] for x, _, _ in self.cross_pt_data]

        # Target data PCA (on alignment window)
        t_data = self.get_time_range(X.shape[1])
        align_idx = self.get_time_idx(t_data, self.tw_align)
        X_tar_r = X[:, align_idx, :].reshape(-1, X.shape[-1])
        tar_dr = self.dim_red(n_components=self.n_comp)
        X_tar_dr = tar_dr.fit_transform(X_tar_r)
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1, X_tar_dr.shape[-1])
        self.tar_dr = tar_dr

        # Choose labels for alignment
        y_align = self.get_align_labels()
        if y_align is None:
            y_align = y

        # Learn alignments to target
        self.algns = []
        for i, info in cache.items():
            algn = self.aligner()
            algn.fit(X_tar_dr, info["X_align"], y_align, info["y_align"])
            self.algns.append(algn)

        # Now do decoding window for each patient
        decode_idx = self.get_time_idx(t_data, self.tw_decode)
        X_tar_r = X[:, decode_idx, :].reshape(-1, X.shape[-1])
        X_tar_dr = self.tar_dr.transform(X_tar_r)
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1, X_tar_dr.shape[-1])

        X_algn_dr = []
        for (i, info), algn in zip(cache.items(), self.algns):
            # reduce patient data in decoding window using cached PCA
            X_r = self.cross_pt_data[i][0][:, decode_idx, :].reshape(-1, self.cross_pt_data[i][0].shape[-1])
            X_dr = info["pca"].transform(X_r)
            X_dr = X_dr.reshape(cross_pt_trials[i], -1, X_dr.shape[-1])

            # apply alignment
            X_algn_dr.append(algn.transform(X_dr).reshape(X_dr.shape[0], -1))

        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

        # Pool datasets
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [info["y"] for info in cache.values()])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([info["y"] for info in cache.values()])

        return X_pool, y_pool

    # -------------------------------------------------------
    # Test-time preprocessing
    # -------------------------------------------------------
    def preprocess_test(self, X):
        t_data = self.get_time_range(X.shape[1])
        decode_idx = self.get_time_idx(t_data, self.tw_decode)
        X_r = X[:, decode_idx, :].reshape(-1, X.shape[-1])
        X_dr = self.tar_dr.transform(X_r)
        return X_dr.reshape(X.shape[0], -1)

    # -------------------------------------------------------
    # Utilities
    # -------------------------------------------------------
    def get_time_range(self, n_pts):
        return np.linspace(self.tw_full[0], self.tw_full[1], n_pts)

    def get_time_idx(self, t_data, tw):
        return np.where((t_data >= tw[0]) & (t_data <= tw[1]))[0]

    def get_align_labels(self):
        return self.y_align

    def set_align_labels(self, y_align):
        self.y_align = y_align