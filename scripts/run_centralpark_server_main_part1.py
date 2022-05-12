from classo import classo_problem

# training and test data for one fold


def one_fold_part1(X_df, X, y, tr, te):

    X_tr = X_df.to_numpy().astype('float').T[tr]
    X_tr /= X_tr.sum(axis=1)[:, None]

    X_te = X_df.to_numpy().astype('float').T[te]
    X_te /= X_te.sum(axis=1)[:, None]

    """
    baseline
    """
    estimator = DummyRegressor()
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_baseline, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=6, verbose=0)
    gscv.fit(X_tr, y[tr])
    yhat = gscv.predict(X_te)
    test_score_baseline = np.mean((yhat-y[te])**2)
    print(f"* Done baseline: {test_score_baseline}.")

    """
    SVR with RBF
    """
    estimator = SVR()
    param_grid_svr_rbf = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': get_rbf_bandwidth(g1)
    }
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_svr_rbf, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=6, verbose=0)
    gscv.fit(X_tr, y[tr])
    print(gscv.best_estimator_)
    yhat = gscv.predict(X_te)
    test_score_svr = np.mean((yhat-y[te])**2)
    print(f"* Done SVC with RBF: {test_score_svr}.")

    """
    Lasso
    """
    estimator = Lasso(random_state=2022)
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_lasso, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=6, verbose=0)
    gscv.fit(X_tr, y[tr])
    print(gscv.best_estimator_)
    yhat = gscv.predict(X_te)
    test_score_lasso = np.mean((yhat-y[te])**2)
    print(f"* Done Lasso: {test_score_lasso}.")

    """
    c-lasso
    """
    problem = classo_problem(X[tr], y[tr], label=label)
    # problem.formulation.w = 1 / nleaves
    problem.formulation.intercept = True
    problem.formulation.concomitant = False
    problem.model_selection.StabSel = False
    problem.model_selection.PATH = True
    problem.model_selection.CV = True
    # one could change logscale, Nsubset, oneSE
    problem.model_selection.CVparameters.seed = (6)
    problem.solve()
    alpha = problem.solution.CV.refit
    yhat = X[te].dot(alpha[1:]) + alpha[0]
    test_score_classo = np.mean((yhat-y[te])**2)
    print(f"* Done classo: {test_score_classo}.")

    """
    RF
    """
    estimator = RandomForestRegressor(max_depth=np.sqrt(X.shape[0]))
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_rf, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=6, verbose=0)
    gscv.fit(X_tr, y[tr])
    test_score_rf = np.mean((gscv.predict(X_te)-y[te])**2)
    print(f"* Done rf: {test_score_rf}.")

    return test_score_baseline, test_score_svc, test_score_lr, test_score_classo, test_score_rf
