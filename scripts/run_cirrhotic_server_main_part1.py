"""
baseline
"""
estimator = DummyClassifier(strategy="stratified", random_state=2022)
# gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_baseline, cv=5,
#                     scoring="accuracy", n_jobs=6, verbose=0)
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
estimator.fit(X_tr, y[tr])
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
yhat = estimator.predict(X_te)
test_score_baseline = np.mean(yhat == y[te])
print(f"* Done baseline: {test_score_baseline}.")

"""
SVC with RBF
"""
estimator = SVC(random_state=2022)
param_grid_svc_rbf = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': get_rbf_bandwidth(g1)
}
gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_svc_rbf, cv=5,
                    scoring="accuracy", n_jobs=6, verbose=0)
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
gscv.fit(X_tr, y[tr])
print(gscv.best_estimator_)
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
yhat = gscv.predict(X_te)
test_score_svc = np.mean(yhat == y[te])
print(f"* Done SVC with RBF: {test_score_svc}.")

"""
Logistic regression
"""
estimator = LogisticRegression(
    penalty='elasticnet', l1_ratio=1, solver='saga', random_state=2022)
gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_lr, cv=5,
                    scoring="accuracy", n_jobs=6, verbose=0)
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
gscv.fit(X_tr, y[tr])
print(gscv.best_estimator_)
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
yhat = gscv.predict(X_te)
test_score_lr = np.mean(yhat == y[te])
print(f"* Done LogisticRegression: {test_score_lr}.")

"""
c-lasso
"""
problem = classo_problem(X[tr], y[tr], label=label)
problem.formulation.intercept = True
problem.formulation.concomitant = False
problem.formulation.classification = True
problem.model_selection.StabSel = False
problem.model_selection.PATH = True
problem.model_selection.CV = True
# one could change logscale, Nsubset, oneSE
problem.model_selection.CVparameters.seed = (6)
problem.solve()
alpha = problem.solution.CV.refit
yhat = X[te].dot(alpha[1:]) + alpha[0]
yhat = np.array([1 if yy > 0 else -1 for yy in yhat])
test_score_classo = np.mean(yhat == y[te])
# selected_idx = problem.solution.CV.selected_param[1:]
# selected_features_classo.append(label[selected_idx])
print(f"* Done classo: {test_score_classo}.")

"""
RF
"""
estimator = RandomForestClassifier(
    max_depth=np.sqrt(X.shape[0]), random_state=2022)
gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_rf, cv=5,
                    scoring='accuracy', n_jobs=6, verbose=0)
# TODO: maybe (also) compare with no pre-screening
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
gscv.fit(X_tr, y[tr])
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
yhat = gscv.predict(X_te)
test_score_rf = np.mean(yhat == y[te])
# selected_idx = np.argsort(
#     -gscv.best_estimator_.feature_importances_)[:n_top]
# selected_features_rf.append(label[selected_idx])
print(f"* Done rf: {test_score_rf}.")
