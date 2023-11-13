import numpy as np
from scipy.linalg import svd, norm
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import laplacian

def RSDE(X, k, n_components=None, n_neighbors=5, lambda_=10e-3, max_iter=1000):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_components is None:
        n_components = k

    assert n_components <= n_samples

    # We used the heat kernel and l2 distance, KNNneighborhood mode with K = 5,
    # and we set the width of the neighborhood σ = 1.

    # COMPUTE SIMILARITY
    kernel = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=True, n_jobs=-1, metric="l2")
    W = laplacian(kernel, normed=True).A
    # W = 0.5 * (knn + knn.T).A

    # INIT B
    B, _, _, = svd(W[:, :1000])
    B = B[:, :k]
    assert B.shape == (n_samples, k)

    # INIT M
    M, _, _, = svd(W[:, :1000])
    M = M[:, :k]
    assert M.shape == (n_samples, k)

    # INIT Q
    Q, _, _, = svd(W[:k, :100])
    Q = Q[:, :k]
    assert Q.shape == (k, k)

    objectives = []
    
    for iter_ in range(max_iter):
        # UPDATE G
        distances = cdist(B, Q)
        argmin = distances.argmin(axis=1)
        G = np.equal.outer(argmin, np.unique(argmin)).astype(int)
        if G.shape[1] < k:
            comp_G = np.zeros((n_samples, k-G.shape[1]))
            G = np.hstack([G, comp_G])
        assert G.shape == (n_samples, k)

        # UPDATE B
        H = (W.T @ M) + (lambda_ * G @ Q)
        U, _, Vt = svd(H, full_matrices=False)
        B = U @ Vt
        assert B.shape == (n_samples, k)

        # UPDATE M
        M = W.T @ B

        # UPDATE Q
        gtb = G.T @ B
        U, _, Vt = svd(gtb, full_matrices=False)
        Q = U @ Vt
        assert Q.shape == (k, k)
        
        objective = norm(W  - (B @ M.T), ord='fro') + lambda_ * norm(B  - (G @ Q), ord='fro')
        objectives.append(objective)
        
    labels = np.argmax(G, axis=1) 
    B = B[:, :n_components]
    
    return labels, B, M, objectives