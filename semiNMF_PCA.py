import numpy as np
from scipy.linalg import eig, inv, svd, norm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph

def RF_semi_nmf_pca(X, k, n_components, n_neighbors=10, alpha=10, max_iter=20, tol = 1e-6, init_spherical=False):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    assert n_components <= n_samples

    # COMPUTE B
    knn = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=True, n_jobs=-1)
    W = 0.5 * (knn + knn.T).A
    D = np.diag(W.sum(axis=0))
    L = D - W
    _, B = eig(L)
    B = B[:, :k]
    assert B.shape == (n_samples, k)

    # INIT G
    if init_spherical:
        from coclust.clustering import SphericalKmeans
        spherical_model = SphericalKmeans(k, random_state=42, n_init=1)
        spherical_model.fit(X)
        km_labels = spherical_model.labels_
    else:
        km_labels = KMeans(k, random_state=42, n_init=1).fit(X).labels_

    G = np.equal.outer(km_labels, np.unique(km_labels)).astype(int)

    if issparse(X):
        X = X.A

    # INIT Q
    Q, _, _, = svd(X.T[:, :1000])
    Q = Q[:, :n_components]
    assert Q.shape == (n_features, n_components)

    # INIT Q_r
    Q_g, _, _, = svd(X.T[:k, :100])
    Q_g = Q_g[:, :k]
    assert Q_g.shape == (k, k)

    # INIT OBJECTIVE
    current_objective = +np.inf
    objectives = []

    # LOOP
    for iter_ in range(max_iter):
        # UPDATE S
        XQ = np.dot(X, Q)
        S = inv((G.T @ G)) @ G.T @ XQ
        assert S.shape == (k, n_components)

        # UPDAGE G
        Bg_tild = B @ Q_g
        distances = cdist(XQ, S)
        distances = distances - (2 * alpha * Bg_tild)
        argmin = distances.argmin(axis=1)
        G = np.equal.outer(argmin, np.unique(argmin)).astype(int)
        assert G.shape == (n_samples, k)

        # UPDATE Q
        H = G @ S
        xth = X.T @ H
        U, _, Vt = svd(xth, full_matrices=False)
        Q = U @ Vt
        assert Q.shape == (n_features, n_components)

        # UPDATE Q_g
        gtb = G.T @ B
        U_g, _, Vt_g = svd(gtb, full_matrices=False)
        Q_g = U_g @ Vt_g
        assert Q_g.shape == (k, k)
    
        objective = norm(X - (H @ Q.T), ord='fro')
        objectives.append(objective)

        if (current_objective - objective) <= tol:
            break
            
        current_objective = objective

    labels = np.argmax(G, axis=1) 

    return labels, S, Q, objectives