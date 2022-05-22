import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import KernelCenterer


def Phi(X_new, X_old, kernel_with_params, center=False, pc=0, return_mean=False):
    """
    Projection on the first `pc` number of PCs. 
    """
    K_old = kernel_with_params(X_old, X_old)
    transformer = KernelCenterer().fit(K_old)
    K_old_tilde = transformer.transform(K_old) if center else K_old
    w_old, V_old = la.eig(K_old_tilde)
    w_old = np.real(w_old)
    V_old = np.real(V_old)
    idx = w_old.argsort()[::-1]
    w_old = w_old[idx]
    V_old = V_old[:, idx]
    K_new = kernel_with_params(X_new, X_old)
    K_new_tilde = transformer.transform(K_new) if center else K_new
    if return_mean:
        return K_new_tilde.dot(V_old[:, pc]).mean(axis=0)/np.sqrt(w_old[pc])
    else:
        return K_new_tilde.dot(V_old[:, pc])/np.sqrt(w_old[pc])
