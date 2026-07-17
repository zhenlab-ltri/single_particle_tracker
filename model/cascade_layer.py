import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split


class CDForestCascadeLayer:
    """One layer of the CDForest cascade, combining four forest estimators via CDUM-based uncertainty weighting.

    Args:
        n_estimators (int, optional): Number of trees per forest. Defaults to 50.
        max_depth (int, optional): Maximum tree depth. Defaults to 7.
        epsilon (float, optional): Significance level. Defaults to 0.05.
        lambda_param (float, optional): Lambda in CDUM = lambda*phi + (1-lambda)*psi. Defaults to 0.9.
        theta (float, optional): Quantile threshold parameter. Defaults to 0.75.
    """

    def __init__(self, n_estimators=50, max_depth=7,
                 epsilon=0.05, lambda_param=0.9, theta=0.75):
        """Initialise a single CDForest cascade layer with four forest estimators.

        Args:
            n_estimators (int, optional): Number of trees per forest. Defaults to 50.
            max_depth (int, optional): Maximum tree depth. Defaults to 7.
            epsilon (float, optional): Significance level. Defaults to 0.05.
            lambda_param (float, optional): Lambda in CDUM = lambda*phi + (1-lambda)*psi. Defaults to 0.9.
            theta (float, optional): Quantile threshold parameter. Defaults to 0.75.

        Returns:
            None
        """
        self.epsilon      = epsilon
        self.lambda_param = lambda_param
        self.theta        = theta

        self.estimators = [
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=10, class_weight="balanced", n_jobs=-1),
            RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=20, class_weight="balanced", n_jobs=-1),
            ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=30, class_weight="balanced", n_jobs=-1),
            ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=44, class_weight="balanced", n_jobs=-1),
        ]
        self.T               = len(self.estimators)
        self.cdum_values     = None
        self.normalized_cdum = None
        self.quantile_thresh = None
        self.cal_alphas      = None
        self.is_trained      = False

    def _safe_proba(self, est, X):
        """Return a two-column probability matrix from an estimator, padding if needed.

        Args:
            est (BaseEstimator): A fitted scikit-learn estimator exposing predict_proba.
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, 2), dtype float32.
        """
        p = est.predict_proba(X)
        if p.shape[1] < 2:
            out = np.zeros((p.shape[0], 2), dtype=np.float32)
            out[:, :p.shape[1]] = p
            return out
        return p.astype(np.float32)

    def _conformity_scores(self, est, X, y_label):
        """Compute conformity scores for class y_label.

        Args:
            est (BaseEstimator): A fitted scikit-learn estimator exposing predict_proba.
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y_label (int): Target class (0 or 1) for which to compute scores.

        Returns:
            np.ndarray: Conformity scores of shape (n_samples,), dtype float32.
        """
        probs     = self._safe_proba(est, X)
        f_y       = probs[:, y_label]
        mask      = np.ones(2, dtype=bool); mask[y_label] = False
        max_other = probs[:, mask].max(axis=1)
        return (1.0 + f_y - max_other) / 2.0

    def _cicp_pvalue(self, alpha_test, cal_alphas_y, rng):
        """Compute class-conditional inductive conformal p-values.

        Args:
            alpha_test (np.ndarray): Conformity scores for test points, shape (n_test,).
            cal_alphas_y (np.ndarray): Calibration conformity scores for class y, shape (n_cal,).
            rng (np.random.Generator): NumPy random generator for drawing tau.

        Returns:
            np.ndarray: p-values of shape (n_test,), dtype float32.
        """
        if len(cal_alphas_y) == 0:
            return np.zeros(len(alpha_test), dtype=np.float32)
        n_cal       = len(cal_alphas_y)
        sorted_cal  = np.sort(cal_alphas_y)
        n_less      = np.searchsorted(sorted_cal, alpha_test, side='left').astype(np.float32)
        n_equal     = (np.searchsorted(sorted_cal, alpha_test, side='right') -
                       np.searchsorted(sorted_cal, alpha_test, side='left')).astype(np.float32)
        tau = rng.uniform(0.0, 1.0, size=len(alpha_test)).astype(np.float32)
        return (n_less + tau * n_equal) / (n_cal + 1.0)

    def _prediction_set(self, p_values, epsilon):
        """Compute the conformal prediction set for each sample.

        Args:
            p_values (np.ndarray): Per-class p-values, shape (n_samples, 2).
            epsilon (float): Significance level controlling set size.

        Returns:
            np.ndarray: Boolean membership matrix of shape (n_samples, 2).
        """
        return p_values > epsilon

    def _compute_cdum(self, est, X_cal, y_cal, X_val, y_val, rng):
        """Compute U(y) = lambda*phi(y) + (1-lambda)*psi(y) for each class y in {0,1}.

        Args:
            est (BaseEstimator): A fitted scikit-learn estimator.
            X_cal (np.ndarray): Calibration feature matrix.
            y_cal (np.ndarray): Calibration labels.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation labels.
            rng (np.random.Generator): NumPy random generator.

        Returns:
            tuple[np.ndarray, dict]: CDUM values of shape (2,) and calibration conformity score dict.
        """
        eps = self.epsilon

        cal_a = {}
        for c in (0, 1):
            mask         = (y_cal == c)
            cal_a[c]     = self._conformity_scores(est, X_cal[mask], c) if mask.any() else np.array([])

        a_val0 = self._conformity_scores(est, X_val, 0)
        a_val1 = self._conformity_scores(est, X_val, 1)
        p0 = self._cicp_pvalue(a_val0, cal_a[0], rng)
        p1 = self._cicp_pvalue(a_val1, cal_a[1], rng)

        psets = self._prediction_set(np.stack([p0, p1], axis=1), eps)

        cdum = np.zeros(2, dtype=np.float32)
        for y in (0, 1):
            vy_mask  = (y_val == y)
            vny_mask = ~vy_mask

            if vy_mask.any():
                gamma_vy = psets[vy_mask]
                gamma_size = gamma_vy.sum(axis=1)

                in_set  = gamma_vy[:, y].astype(float)
                rho     = np.where(gamma_size > 0,
                                   (gamma_size - in_set).astype(float),
                                   1.0)
                phi_y = float(rho.mean())
            else:
                phi_y = 1.0

            if vny_mask.any():
                psi_y = float(psets[vny_mask, y].mean())
            else:
                psi_y = 1.0

            cdum[y] = self.lambda_param * phi_y + (1.0 - self.lambda_param) * psi_y

        return cdum, cal_a

    def fit_and_calibrate(self, X, y):
        """Train all T forests, compute CDUM per forest, and build normalized CDUM and quantile thresholds.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Label array of shape (n_samples,).

        Returns:
            bool: True if training succeeded, False otherwise.
        """
        if len(np.unique(y)) < 2:
            return False

        classes, counts = np.unique(y, return_counts=True)
        if counts.min() < 5:
            min_cls = classes[counts.argmin()]
            min_idx = np.where(y == min_cls)[0]
            extra   = np.random.choice(min_idx, 5 - len(min_idx), replace=True)
            X = np.vstack([X, X[extra]])
            y = np.concatenate([y, y[extra]])

        def safe_split(X, y, **kw):
            """Perform a stratified train/test split, falling back to unstratified on failure.

            Args:
                X (np.ndarray): Feature matrix.
                y (np.ndarray): Label array.
                **kw: Additional keyword arguments forwarded to train_test_split.

            Returns:
                tuple: Split arrays (X_train, X_test, y_train, y_test).
            """
            c, cnt = np.unique(y, return_counts=True)
            strat  = y if cnt.min() >= 2 else None
            try:    return train_test_split(X, y, stratify=strat, **kw)
            except: return train_test_split(X, y, **kw)

        X_tr, X_rest, y_tr, y_rest = safe_split(X, y, test_size=0.2, random_state=42)
        X_cal, X_val, y_cal, y_val = safe_split(X_rest, y_rest, test_size=0.5, random_state=42)

        rng = np.random.default_rng(seed=0)

        self.cdum_values = np.zeros((self.T, 2), dtype=np.float32)
        self.cal_alphas  = [{} for _ in range(self.T)]

        for t, est in enumerate(self.estimators):
            est.fit(X_tr, y_tr)
            cdum_t, cal_a_t       = self._compute_cdum(est, X_cal, y_cal, X_val, y_val, rng)
            self.cdum_values[t]   = cdum_t
            self.cal_alphas[t]    = cal_a_t

        col_sum              = self.cdum_values.sum(axis=0) + 1e-8
        self.normalized_cdum = self.cdum_values / col_sum

        q = np.ceil((self.T + 1) * self.theta) / self.T * 100.0
        q = float(np.clip(q, 0.0, 100.0))
        self.quantile_thresh = np.percentile(self.normalized_cdum, q, axis=0,
                                             method='lower')

        self.is_trained = True
        return True

    def transform_and_filter(self, X):
        """Apply layer-wise uncertainty filtering and return the augmented feature representation.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Filtered output of shape (n_samples, 2*T).
        """
        if not self.is_trained:
            return np.zeros((X.shape[0], 2 * self.T), dtype=np.float32)
        outputs = []
        for t, est in enumerate(self.estimators):
            p = self._safe_proba(est, X).copy()
            if self.normalized_cdum[t, 0] > self.quantile_thresh[0]:
                p[:, 0] = 0.0
            if self.normalized_cdum[t, 1] > self.quantile_thresh[1]:
                p[:, 1] = 0.0
            outputs.append(p)
        return np.hstack(outputs)

    def inference_weights(self, y_label):
        """Compute per-forest inference weights for the given class label (Eq. 14).

        Args:
            y_label (int): Target class (0 or 1).

        Returns:
            np.ndarray: Normalised weight array of shape (T,), dtype float32.
        """
        if not self.is_trained:
            return np.ones(self.T, dtype=np.float32) / self.T
        u = self.cdum_values[:, y_label]
        w = (1.0 - u) / (1.0 + u + 1e-8)
        w = np.maximum(w, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.ones(self.T) / self.T

    def conformal_pvalues(self, X, rng):
        """Compute per-forest conformal p-values for aggregation (Eq. 18).

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            rng (np.random.Generator): NumPy random generator.

        Returns:
            np.ndarray: p-value tensor of shape (n_samples, T, 2), dtype float32.
        """
        if not self.is_trained:
            return np.full((X.shape[0], self.T, 2), 0.5, dtype=np.float32)
        out = np.zeros((X.shape[0], self.T, 2), dtype=np.float32)
        for t, est in enumerate(self.estimators):
            for y in (0, 1):
                a_test     = self._conformity_scores(est, X, y)
                cal_a_y    = self.cal_alphas[t].get(y, np.array([]))
                out[:, t, y] = self._cicp_pvalue(a_test, cal_a_y, rng)
        return out
