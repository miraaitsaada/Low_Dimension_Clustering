import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale as scale_

def _init(X, k, q, seed, init_kmeans, whiten_init):
    n, d = X.shape

    if scipy.sparse.issparse(X):
        X = X.A
    
    Q, s, Pt = scipy.linalg.svd(X, full_matrices=False)
    A = Pt[:q].T
    
    if whiten_init:
        A = A / s[:q]
        
    XA = X @ A
    
    if init_kmeans:
        init_labels = KMeans(k, random_state=seed).fit(XA).labels_
    else:
        init_labels = np.random.RandomState(seed).choice(k, size=n)
    
    U = np.eye(k)[init_labels]
    F = ((U.T[:, np.newaxis] @ XA).squeeze(1).T / U.sum(axis=0)).T
    
    return U, A, F

def RKM(X, k, seed, q, max_iter=20, tol=1e-6, scale=False, init_kmeans=True, whiten=False, whiten_init=False):
    n, d = X.shape
    curr_obj = -np.inf
    objs = []

    if scale:
        X = scale_(X, with_mean=True, with_std=True)
    
    # init
    U, A, F = _init(X, k, q, seed, init_kmeans, whiten_init)
    
    for i in range(max_iter):
        Q, s, Pt = scipy.linalg.svd((U @ F).T @ X)
        Q = Q[:, :q]
        P = Pt[:q].T
        s = s[:q]

        # update A
        A = P @ Q.T
        XA = X @ A
        
        if whiten:
            XA /= np.sqrt(np.var(XA, axis=0))

        # update U
        labels = np.argmin(np.sqrt(((XA - F[:, np.newaxis])**2).sum(axis=2)), axis=0)
        U = np.eye(k)[labels]

        # update F
        F = ((U.T[:, np.newaxis] @ XA).squeeze(1).T / U.sum(axis=0)).T
        # F = np.linalg.inv(U.T @ U) @ U.T @ X @ A

        # criterion
        obj = np.power(np.linalg.norm(X - (U @ F @ A.T)), 2)

        objs.append(obj)

        if np.abs(curr_obj - obj) <= tol:
            break

        curr_obj = obj
        
    return labels, objs

def FKM(X, k, seed, q, max_iter=20, tol=1e-6, scale=False, init_kmeans=True):
    n, d = X.shape
    curr_obj = -np.inf
    objs = []

    if scale:
        X = scale_(X, with_mean=True, with_std=True)

    U, A, F = _init(X, k, q, seed, init_kmeans, whiten_init=False)
    
    for i in range(max_iter):
        # update U
        labels = np.argmin(np.sqrt(((X @ A - F[:, np.newaxis])**2).sum(axis=2)), axis=0)
        U = np.eye(k)[labels]
        
        # update A
        e, V = np.linalg.eig(X.T @ ((U @ ((U.T[:, np.newaxis]).squeeze(1).T / U.sum(axis=0)).T) - 1) @ X)
        A = V[:, :q]
        XA = X @ A

        # update F
        F = ((U.T[:, np.newaxis] @ XA).squeeze(1).T / U.sum(axis=0)).T

        # criterion
        obj = np.power(np.linalg.norm(XA - (U @ F)), 2)

        objs.append(obj)

        if np.abs(curr_obj - obj) <= tol:
            break

        curr_obj = obj
        
    return labels, objs