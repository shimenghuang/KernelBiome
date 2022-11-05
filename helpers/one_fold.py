import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from classo import classo_problem
from kernelbiome.kernelbiome import KernelBiome


def run_bl(X_tr, y_tr, X_te, task_type, random_state=None):
    """
    yscore: between 0 and 1
    yhat: -1 or 1 (yscore <= 0.5) or 1 (yscore > 0.5)

    Note: `np.random.uniform` samples from [low, high), here we ensure that
        0.5 is included in the lower class, which is consistent with
        the rest methods in sklearn
    """
    np.random.seed(random_state)
    if task_type == 'classification':
        pos_class = np.mean(y_tr == 1)
        neg_class = np.mean(y_tr == -1)
        if pos_class < neg_class:
            # yscore = np.random.uniform(0, 0.5, X_te.shape[0])
            yscore = -np.random.uniform(-0.5, 0, X_te.shape[0])
            yhat = np.repeat(-1, X_te.shape[0])
        else:
            # yscore = np.random.uniform(0.5, 1, X_te.shape[0])
            yscore = -np.random.uniform(-1, -0.5, X_te.shape[0])
            yhat = np.repeat(1, X_te.shape[0])
    else:
        yscore = np.repeat(np.mean(y_tr), X_te.shape[0])
        yhat = yscore
    return yscore, yhat


def run_svm_rbf(X_tr, y_tr, X_te, task_type, scoring,
                param_grid=None, random_state=None, n_jobs=-1):
    """
    yscore: between 0 and 1
    yhat: -1 (yscore <= 0.5) or 1 (yscore > 0.5)
    """
    if task_type == 'classification':
        SVM = SVC(random_state=random_state, probability=True)
    else:
        SVM = SVR()

    if param_grid is None:
        estimator = SVM()
    else:
        estimator = GridSearchCV(estimator=SVM, param_grid=param_grid, cv=5,
                                 scoring=scoring, n_jobs=n_jobs, verbose=0)
    estimator.fit(X_tr, y_tr)

    if task_type == 'classification':
        yscore = estimator.predict_proba(X_te)[:, 1]
        yhat = estimator.predict(X_te)
    else:
        yscore = estimator.predict(X_te)
        yhat = yscore
    return yscore, yhat


def run_lr_l1(X_tr, y_tr, X_te, task_type, scoring, param_grid=None,
              random_state=None, n_jobs=-1):
    """
    yscore: between 0 and 1
    yhat: -1 (yscore <= 0.5) or 1 (yscore > 0.5)
    """
    if task_type == 'classification':
        LR = LogisticRegression(penalty='l1',
                                solver='liblinear',
                                random_state=random_state)
    else:
        LR = Lasso(random_state=random_state)

    if param_grid is None:
        estimator = LR
    else:
        estimator = GridSearchCV(estimator=LR, param_grid=param_grid, cv=5,
                                 scoring=scoring, n_jobs=n_jobs, verbose=0)
    estimator.fit(X_tr, y_tr)

    if task_type == 'classification':
        yscore = estimator.predict_proba(X_te)[:, 1]
        yhat = estimator.predict(X_te)
    else:
        yscore = estimator.predict(X_te)
        yhat = yscore
    return yscore, yhat


def run_rf(X_tr, y_tr, X_te, task_type, scoring, param_grid=None,
           random_state=None, n_jobs=-1):
    """
    yscore: between 0 and 1
    yhat: -1 (yscore <= 0.5) or 1 (yscore > 0.5)
    """
    if task_type == 'classification':
        RF = RandomForestClassifier(n_estimators=500,
                                    random_state=random_state)
    else:
        RF = RandomForestRegressor(n_estimators=500,
                                   random_state=random_state)

    if param_grid is None:
        estimator = RF
    else:
        estimator = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5,
                                 scoring=scoring, n_jobs=n_jobs, verbose=0)
    estimator.fit(X_tr, y_tr)

    if task_type == 'classification':
        yscore = estimator.predict_proba(X_te)[:, 1]
        yhat = estimator.predict(X_te)
    else:
        yscore = estimator.predict(X_te)
        yhat = yscore
    return yscore, yhat


def run_classo(X_tr, y_tr, X_te, task_type, random_state=None):
    """
    yscore: real numbers
    yhat: -1 (yscore <= 0) or 1 (yscore > 0)
    """
    pseudo_count = np.min(X_tr[np.where(X_tr != 0)])/2
    X_tr_cl = np.log(X_tr + pseudo_count)
    problem = classo_problem(X_tr_cl, y_tr)
    problem.formulation.intercept = True
    problem.formulation.concomitant = False
    problem.model_selection.StabSel = False
    problem.model_selection.PATH = True
    problem.model_selection.CV = True
    if task_type == 'classification':
        problem.formulation.classification = True
    else:
        problem.formulation.classification = False
    problem.model_selection.CVparameters.seed = (random_state)  # 6
    problem.solve()
    alpha = problem.solution.CV.refit

    X_te_cl = np.log(X_te + pseudo_count)
    yscore = X_te_cl.dot(alpha[1:]) + alpha[0]
    if task_type == 'classification':
        yhat = (yscore > 0)*2 - 1
    else:
        yhat = yscore
    return yscore, yhat


def run_kb(X_tr, y_tr, X_te, task_type, scoring, param_grid=None,
           models=None, outer_cv_type=None, grp=None,
           n_hyper_grid=10, random_state=None, n_jobs=-1, outpath=None):
    """
    yscore: between 0 and 1
    yhat: -1 (yscore <= 0.5) or 1 (yscore > 0.5)
    """
    # Setup parameters
    cv_pars = {'outer_cv_type': outer_cv_type,
               'grp': grp,
               'scoring': scoring,
               }
    estimator_pars = {'n_hyper_grid': n_hyper_grid}
    if task_type == 'classification':
        kernel_estimator = "SVC"
        center_kmat = True
        if cv_pars['outer_cv_type'] is None:
            cv_pars['outer_cv_type'] = "stratified"
        estimator_pars['probability_refit'] = True
        estimator_pars['max_iter'] = 50000
        estimator_pars['cache_size'] = 1000
    else:
        kernel_estimator = "KernelRidge"
        center_kmat = True
        if cv_pars['outer_cv_type'] is None:
            cv_pars['outer_cv_type'] = "kfold"

    # Fit KernelBiome
    KB = KernelBiome(kernel_estimator=kernel_estimator,
                     center_kmat=center_kmat,
                     hyperpar_grid=None,
                     models=models,
                     cv_pars=cv_pars,
                     estimator_pars=estimator_pars,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     verbose=1)
    KB.fit(X_tr, y_tr)
    if outpath is not None:
        KB.best_models_.to_csv(outpath, index=False)

    # Compute scores
    if task_type == 'classification':
        yscore = KB.predict_proba(X_te, X_tr)[:, 1]
        yhat = KB.predict(X_te)
    else:
        yscore = KB.predict(X_te)
        yhat = yscore

    return yscore, yhat
