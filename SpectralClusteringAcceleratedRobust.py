import numpy as np
import scipy.sparse as sp

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

from Nystrom import eigendecompositionNystrom



class SCAR:
    """
    Implementation of SCAR from the paper
    'SCAR — Spectral Clustering Accelerated and Robustified'
    
    This code is based on the implementation of Robust Spectral Clustering by:
    Aleksandar Bojchevski, Yves Matkovic, and Stephan Günnemann.
    2017. Robust Spectral Clustering for Noisy Data.
    In Proceedings of KDD’17, August 13–17, 2017, Halifax, NS, Canada.
    
    Copyright (C) 2022 SpectralClusteringAcceleratedRobust
    
    """

    def __init__(self, k, nn, alpha, theta=20, m=0.5, laplacian=0, n_iter=50, normalize=False, weighted=False, verbose=False, seed=0):
        """
        :param k: number of clusters
        :param nn: number of neighbours to consider for constructing the KNN graph (excluding the node itself)
        :param alpha: percentage of landmark points selected as subsample for the Nyström method from the original dataset
        :param theta: number of corrupted edges to remove
        :param m: minimum percentage of neighbours to keep per node (omega_i constraints)
        :param laplacian: which graph Laplacian to use: 0: L, 1: L_rw, 2: L_sym
        :param n_iter: number of iterations of the alternating optimization procedure
        :param normalize: whether to row normalize the eigenvectors before performing k-means
        :param weighted: use weighted (True) or unweighted (False) k-nn as similarity graph
        :param verbose: verbosity
        :param seed: random state
        """

        self.k = k
        self.nn = nn
        self.alpha = alpha
        self.theta = theta
        self.m = m
        self.laplacian = laplacian
        self.n_iter = n_iter
        self.normalize = normalize
        self.weighted = weighted
        self.verbose = verbose
        self.seed = seed

        np.random.seed(self.seed)

        if laplacian == 0:
            if self.verbose:
                print('Using unnormalized Laplacian L')
        elif laplacian == 1:
            if self.verbose:
                print('Using random walk based normalized Laplacian L_rw')
        elif laplacian == 2:
            if self.verbose:
                print('Using symmetric normalized Laplacian L_sym')
        else:
            raise ValueError('Choice of graph Laplacian not valid. Please use 0, 1 or 2.')
    

    def __latent_decomposition(self, X):
        """
        :param X: input datapoints to be clustered
        
        :return Ag: good graph, containing only edges not identified as noise
        :return Ac: corrupted graph, containing all edges identified as noise
        :return H: spectral embedding of the selected graph Laplacian
        """

        # compute weighted k-nn graph
        if self.weighted == True:
            A = kneighbors_graph(X=X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='distance')
            gamma = -1.0/(X.shape[0]*X.shape[1])
            A.data = np.exp(gamma*(A.data**2))
        # compute unweighted k-nn graph
        else:
            A = kneighbors_graph(X=X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='connectivity')
        
        # make the graph undirected
        A = A.maximum(A.T)

        # number of nodes
        N = A.shape[0]
        # node degrees
        deg = A.sum(0).A1

        # check the trace for convergence
        prev_trace = np.inf
        Ag = A.copy()

        for it in range(self.n_iter):
            
            # form the unnormalized Laplacian
            D = sp.diags(Ag.sum(0).A1).tocsc()
            L = D - Ag

            # form random-walk Laplacian
            if self.laplacian == 1:
                D_minus = D
                D_minus.data = 1/D.data
                L = D_minus.dot(L)
            
            # form symmetric Laplacian
            elif self.laplacian == 2:
                D_minus_half = D
                D_minus_half.data = 1/(np.sqrt(D.data))
                L = L.dot(D_minus_half)
                L = D_minus_half.dot(L)
                
            elif self.laplacian != 0: 
                raise ValueError("the laplacian {} is not defined".format(self.laplacian))

            # calculate eigendecomposition using the Nyström method
            H, lam = eigendecompositionNystrom(L, self.k, self.alpha)

            trace = lam.sum()

            if self.verbose:
                print('Iter: {} Trace: {:.4f}'.format(it, trace))

            if self.theta == 0:
                # no edges are removed
                Ac = sp.coo_matrix((N, N), [np.int])
                break

            if prev_trace - trace < 1e-10:
                # we have converged
                break
            
            allowed_to_remove_per_node = (deg * self.m).astype(np.int)
            prev_trace = trace
            
            # consider only the edges on the lower triangular part since we are symmetric
            edges = sp.tril(A).nonzero()
            removed_edges = []

            # for random-walk Laplacian
            if self.laplacian == 1:
                # fix for potential numerical instability of the eigenvalues computation
                lam[np.isclose(lam, 0)] = 0

                # refers to equation 3.11 in the thesis
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2 \
                    - np.linalg.norm(H[edges[0]] * np.sqrt(lam), axis=1) ** 2 \
                    - np.linalg.norm(H[edges[1]] * np.sqrt(lam), axis=1) ** 2
            
            # for unnormalized and symmetric Laplacian
            else:
                # refers to equation 3.10 in the thesis
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2

            # greedly remove the worst edges
            for ind in p.argsort()[::-1]:
                e_i, e_j, p_e = edges[0][ind], edges[1][ind], p[ind]
                
                # remove the edge if it satisfies the constraints
                if allowed_to_remove_per_node[e_i] > 0 and allowed_to_remove_per_node[e_j] > 0 and p_e > 0:
                    allowed_to_remove_per_node[e_i] -= 1
                    allowed_to_remove_per_node[e_j] -= 1
                    removed_edges.append((e_i, e_j))
                    if len(removed_edges) == self.theta:
                        break

            removed_edges = np.array(removed_edges)
            
            Ac = sp.coo_matrix((np.ones(len(removed_edges)), (removed_edges[:, 0], removed_edges[:, 1])), shape=(N, N))
            Ac = Ac.maximum(Ac.T)
            Ag = A - Ac
        
        return Ag, Ac, H
    

    def fit_predict(self, X):
        """
        :param X: array-like or sparse matrix, shape (n_samples, n_features)
        
        :return labels: cluster labels ndarray, shape (n_samples)
        """
       
        Ag, Ac, H = self.__latent_decomposition(X)
            
        self.Ag = Ag
        self.Ac = Ac

        # norm row-wise to increase eigenvector accuracy and stability
        if self.normalize:
            self.H = np.nan_to_num(H / np.linalg.norm(H, axis=1)[:, None])
        else:
            self.H = H

        # cluster rows of obtained eigenvectors to obtain clutering labels
        labels = KMeans(n_clusters=self.k, random_state=self.seed).fit_predict(self.H)

        self.labels = labels
        
        return labels
