import numpy as np
import warnings
import timeit
import copy
from kernelbiome.helpers_fitting import (models_to_kernels,
                                         top_model_per_kernel_class,
                                         default_kernel_models,
                                         outer_cv, fit_single_model)
from kernelbiome.helpers_analysis import (modelname_to_fun,
                                          df_ke_dual_mat,
                                          get_cfi, get_cpd)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, SVC
from sklearn.model_selection import (StratifiedKFold,
                                     LeaveOneGroupOut,
                                     KFold)


class KernelBiome:
    """KernelBiome class

    """
    # Initialize KernelBiome

    def __init__(
            self,
            kernel_estimator='SVC',
            center_kmat=False,
            hyperpar_grid=None,
            models=None,
            cv_pars={},
            estimator_pars={},
            n_jobs=1,
            random_state=None,
            verbose=1
    ):
        # Initialize all internal variables
        self.kernel_estimator_ = copy.deepcopy(kernel_estimator)
        self._center_kmat = copy.deepcopy(center_kmat)
        self._hyperpar_grid = copy.deepcopy(hyperpar_grid)
        self._models = copy.deepcopy(models)
        self._cv_pars = copy.deepcopy(cv_pars)
        self._estimator_pars = copy.deepcopy(estimator_pars)
        self._estimator_pars_outer = copy.deepcopy(estimator_pars)
        self._n_jobs = copy.deepcopy(n_jobs)
        self._random_state = copy.deepcopy(random_state)
        self._verbose = copy.deepcopy(verbose)
        # Deal with special inputs
        if 'probability_refit' in self._estimator_pars:
            del self._estimator_pars_outer['probability_refit']
            self._estimator_pars['probability'] = self._estimator_pars.pop(
                'probability_refit')
        if 'n_hyper_grid' in self._estimator_pars:
            del self._estimator_pars_outer['n_hyper_grid']
            self._n_hyper_grid = self._estimator_pars.pop(
                'n_hyper_grid')
        else:
            self._n_hyper_grid = 10
        # Set default parameters for CV
        if 'outer_cv_type' not in self._cv_pars:
            if self.kernel_estimator_ in ['SVR', 'KernelRidge']:
                self._cv_pars['outer_cv_type'] = "kfold"
            else:
                self._cv_pars['outer_cv_type'] = "stratified"
        if 'n_fold_inner' not in cv_pars:
            self._cv_pars['n_fold_inner'] = 5
        if 'n_fold_outer' not in cv_pars:
            self._cv_pars['n_fold_outer'] = 10
        if 'scoring' not in cv_pars:
            if self.kernel_estimator_ in ['SVR', 'KernelRidge']:
                self._cv_pars['scoring'] = 'neg_mean_squared_error'
            else:
                self._cv_pars['scoring'] = 'accuracy'
        # Select correct sklearn estimator
        if self.kernel_estimator_ == "KernelRidge":
            self._estimator = KernelRidge(kernel="precomputed")
        elif self.kernel_estimator_ == "SVR":
            self._estimator = SVR(kernel="precomputed")
        elif self.kernel_estimator_ == "SVC":
            self._estimator = SVC(kernel="precomputed")
        else:
            raise ValueError(
                "kernel_estimator can only be one of" +
                "'KernelRidge', 'SVR', or 'SVC'.")

    # Fit KernelBiome
    def fit(self, X, y, w=None):
        """Fit model"""
        # Ensure data is in the simplex
        if np.abs(np.sum(X.sum(axis=1)) - X.shape[0]) > 10e-5:
            warnings.warn("X does not live on the simplex, rescaling.")
            X /= X.sum(axis=1)[:, None]

        # Generate kernel_dict (if not already passed)
        self.w_ = copy.deepcopy(w)
        if self._models is None:
            kernel_dict = models_to_kernels(
                default_kernel_models(X))
        elif callable(self._models[list(self._models.keys())[0]]):
            kernel_dict = self._models
        else:
            # Check whether weighting is correctly specified if weighted
            # kernel is specified in _models
            for ss in list(self._models.keys()):
                if '-weighted' in ss.split('_', maxsplit=1)[0]:
                    if self.w_ is None:
                        TypeError("Weighted kernels were specified" +
                                  " but weights 'w' are None.")
            kernel_dict = models_to_kernels(self._models, self.w_)

        # Generate outer CV splits
        if self._cv_pars['outer_cv_type'] == "stratified":
            if 'shuffle' not in self._cv_pars:
                self._cv_pars['shuffle'] = True
            kf = StratifiedKFold(n_splits=self._cv_pars['n_fold_outer'],
                                 shuffle=self._cv_pars['shuffle'],
                                 random_state=self._random_state)
            cv_split = list(kf.split(X, y))
        elif self._cv_pars['outer_cv_type'] == "kfold":
            if 'shuffle' not in self._cv_pars:
                self._cv_pars['shuffle'] = True
            kf = KFold(n_splits=self._cv_pars['n_fold_outer'],
                       shuffle=self._cv_pars['shuffle'])
            cv_split = list(kf.split(X, y))
        elif self._cv_pars['outer_cv_type'] == "logo":
            if 'grp' not in self._cv_pars:
                raise TypeError("Missing 'grp' parameter in cv_pars.")
            logo = LeaveOneGroupOut()
            self._cv_pars['n_fold_outer'] = logo.get_n_splits(
                X, y, self._cv_pars['grp'])
            cv_split = list(logo.split(X, y, self._cv_pars['grp']))

        # Iterate over models (outer CV) - only if len(kernel_dict) > 1
        if len(kernel_dict) > 1:
            n_estimator = len(kernel_dict)
            train_scores = np.full((n_estimator,
                                    self._cv_pars['n_fold_outer']), np.nan)
            test_scores = np.full((n_estimator,
                                   self._cv_pars['n_fold_outer']), np.nan)
            selected_params = np.full((n_estimator,
                                       self._cv_pars['n_fold_outer']),
                                      np.nan, dtype=object)
            time_estimators = np.full(n_estimator, np.nan)
            if self._verbose > 0:
                print("Running outer CV...")
            for ii, (name, kmat_fun) in enumerate(kernel_dict.items()):
                if self._verbose > 0:
                    print(f"--- running: {name} ---")
                # Time outer cv
                time_estimators[ii] = timeit.default_timer()
                # Run outer CV
                if self._verbose > 0:
                    lambdas = np.real(np.linalg.eigvals(kmat_fun(X, X)))
                    print('* max/min eigenvalue of K: ' +
                          str(np.max(lambdas)) + '/' + str(np.min(lambdas)))
                cv_results = outer_cv(
                    X, y, self._estimator,
                    kmat_fun=kmat_fun,
                    center_kmat=self._center_kmat,
                    cv_split=cv_split,
                    scoring=self._cv_pars['scoring'],
                    n_fold_outer=self._cv_pars['n_fold_outer'],
                    n_fold_inner=self._cv_pars['n_fold_inner'],
                    hyperpar_grid=self._hyperpar_grid,
                    n_hyper_grid=self._n_hyper_grid,
                    estimator_pars=self._estimator_pars_outer,
                    n_jobs=self._n_jobs,
                    verbose=self._verbose-1)
                train_scores[ii, :] = cv_results['train_scores']
                test_scores[ii, :] = cv_results['test_scores']
                selected_params[ii, :] = cv_results['selected_params']
                if self._verbose > 0:
                    print(
                        f"* average test score: {np.mean(test_scores[ii, :])}")
                # Time outer cv
                time_estimators[ii] = timeit.default_timer() - \
                    time_estimators[ii]
            # Collect results
            outer_cv_results = {'kernel_dict': kernel_dict,
                                'time_estimators': time_estimators,
                                'train_scores': train_scores,
                                'test_scores': test_scores}
            # Select best model
            best_models = top_model_per_kernel_class(
                kernel_dict,
                train_scores,
                test_scores,
                selected_params)
            model_selected = best_models.iloc[0]
            if self._verbose > 0:
                print(f"best model is {model_selected.estimator_key}")
        else:
            if self._verbose > 0:
                print(
                    "Not running outer CV since only one model was provided.")
            outer_cv_results = None
            best_models = None
            model_selected = {'estimator_key': list(kernel_dict.keys())[0],
                              'kmat_fun': list(kernel_dict.values())[0]}
        # Refit best model on full data
        if self._verbose > 0:
            print("Refit best model")
        fitted_mod = fit_single_model(
            X, y,
            estimator=self._estimator,
            kmat_fun=model_selected['kmat_fun'],
            scoring=self._cv_pars['scoring'],
            hyperpar_grid=self._hyperpar_grid,
            center_kmat=self._center_kmat,
            return_gscv=False,
            n_hyper_grid=self._n_hyper_grid,
            estimator_pars=self._estimator_pars,
            n_fold=self._cv_pars['n_fold_inner'],
            n_jobs=self._n_jobs,
            verbose=self._verbose-1)
        train_score = np.mean(fitted_mod['train_score'])

        # Combine model_selected and fitted_mod dicts
        fitted_mod['kmat_fun'] = model_selected[
            'kmat_fun']
        fitted_mod['estimator_key'] = model_selected[
            'estimator_key']
        if len(kernel_dict) > 1:
            fitted_mod['avg_CV_score'] = model_selected[
                'avg_test_score']
            fitted_mod['most_freq_hyperpar_CV'] = model_selected[
                'most_freq_best_param']

        # Collect output
        self.outer_cv_results_ = outer_cv_results
        self.best_models_ = best_models
        self.fitted_model_ = fitted_mod
        self.train_score_ = train_score

    # Predict function
    def predict(self, X):
        return self.fitted_model_['pred_fun'](X)

    # Predict probabilities
    def predict_proba(self, Xnew, Xtrain):
        if self.kernel_estimator_ != 'SVC':
            TypeError(
                "predict_proba only exists for kernel_estimator == 'SVC'")
        K = self.fitted_model_['kmat_fun'](Xnew, Xtrain)
        if self.fitted_model_['transformer'] is not None:
            K = self.fitted_model_['transformer'].transform(K)
        return self.fitted_model_['estimator'].predict_proba(K)

    # Compute CFI
    def compute_cfi(self, Xtrain, Xnew=None, verbose=0):
        if Xnew is None:
            Xnew = Xtrain
        estimator_name = self.kernel_estimator_
        # Select sample support and coefs depending on estimator
        if estimator_name == 'KernelRidge':
            idx_supp = np.array(range(Xtrain.shape[0]))
            coefs = self.fitted_model_['estimator'].dual_coef_
        else:
            idx_supp = self.fitted_model_['estimator'].support_
            coefs = self.fitted_model_['estimator'].dual_coef_[0]
        k_fun = modelname_to_fun(self.fitted_model_['estimator_key'],
                                 w=self.w_)
        if k_fun is None:
            TypeError("Kernel of best model does not exist.")
        # Compute derivative of fitted kernel estimator
        if self.fitted_model_['transformer'] is not None:
            center_kmat = True
        else:
            center_kmat = False
        df = df_ke_dual_mat(Xnew, Xtrain, center_kmat,
                            coefs, idx_supp, k_fun, verbose)
        cfi_vals = get_cfi(Xnew, df, proj=True)
        return cfi_vals

    # Compute CPD
    def compute_cpd(self, Xtrain, evaluation_grid=None,
                    comp_idx=None, rescale=True):
        if evaluation_grid is None:
            evaluation_grid = Xtrain
        estimator_name = self.kernel_estimator_
        if estimator_name not in ['SVR', 'KernelRidge']:
            raise TypeError("CPDs are only implemented for regression." +
                            "Estimator needs to be 'SVR' or 'KernelRidge'.")
        cpd_vals = get_cpd(Xtrain, evaluation_grid,
                           self.fitted_model_['pred_fun'],
                           comp_idx, rescale, verbose=False)
        return(cpd_vals)

    # Compute kernel principle components
    def kernelPCA(self, Xtrain, Xnew=None, num_pc=2):
        """kernelPCA

        Project Xnew onto the first `num_pc` number of PCs and compute
        for each of then used PCs how much each component contributes.
        """
        if Xnew is None:
            Xnew = Xtrain
        Kmat = self.fitted_model_['kmat_fun'](Xtrain, Xtrain)
        if self.fitted_model_['transformer'] is not None:
            Kmat = self.fitted_model_['transformer'].transform(Kmat)
        wtrain, Vtrain = np.linalg.eig(Kmat)
        wtrain = np.real(wtrain)
        Vtrain = np.real(Vtrain)
        idx = wtrain.argsort()[::-1]
        wtrain = wtrain[idx]
        Vtrain = Vtrain[:, idx]
        Knew = self.fitted_model_['kmat_fun'](Xnew, Xtrain)
        if self.fitted_model_['transformer'] is not None:
            Knew = self.fitted_model_['transformer'].transform(Knew)
        projection = Knew.dot(Vtrain[:, 0:num_pc])/np.sqrt(wtrain[0:num_pc])
        # Compute contribution of each component to the PCs
        coordinate_contribution = np.zeros((Xtrain.shape[1], num_pc))
        for j in range(Xtrain.shape[1]):
            Xj = copy.deepcopy(Xtrain)
            Xj[:, j] *= 1.1
            Xj /= Xj.sum(axis=1)[:, None]
            KK = self.fitted_model_['kmat_fun'](
                Xj, Xtrain)
            if self.fitted_model_['transformer'] is not None:
                KK = self.fitted_model_['transformer'].transform(KK)
            Ktmp = (Knew - KK).mean(axis=0)
            coordinate_contribution[j, :] = Ktmp.dot(
                Vtrain[:, 0:num_pc])/np.sqrt(wtrain[0:num_pc])
        return projection, coordinate_contribution

    # Compute kernel summary statistics
    def kernelSumStat(self, Xtrain, Xnew=None, u=None):
        """Kernel Summary Statistics

        Computes the kernel summary statistic, i.e. the
        kernel distance for a sample point to u, based on most
        predictive kernel.  Note: Not affected by scaling.

        """
        if Xnew is None:
            Xnew = Xtrain
        k_fun = modelname_to_fun(self.fitted_model_['estimator_key'],
                                 w=self.w_)
        if k_fun is None:
            TypeError("Kernel of best model does not exist.")

        # Compute summary statistic
        if u is None:
            u = np.array([1/Xnew.shape[1]]*Xnew.shape[1])
        ku = k_fun(u, u)
        sumstat = np.empty(Xnew.shape[0])
        for kk in range(Xnew.shape[0]):
            sumstat[kk] = (2 * k_fun(Xnew[kk, :], u) -
                           k_fun(Xnew[kk, :], Xnew[kk, :]) - ku)

        return sumstat

    # Compute kernel distance
    def kernelDist(self, X, Y):
        """Kernel Distance

        Computes the kernel distance between X and Y, which should be
        (m1 x d) and (m2 x d), respectively.
        """
        m1, d = X.shape
        m2, d2 = Y.shape
        assert(d == d2)

        k_fun = self.fitted_model_['kmat_fun']

        # Compute distance
        dist_mat = -2 * k_fun(X, Y)
        dX = np.zeros((m1, m2))
        dY = np.zeros((m1, m2))
        for k in range(X.shape[0]):
            dX[k, :] = k_fun(X[k, :][None, :], X[k, :][None, :])
        for k in range(Y.shape[0]):
            dY[:, k] = k_fun(Y[k, :][None, :], Y[k, :][None, :])
        dist_mat += dX + dY

        return np.asarray(dist_mat)
