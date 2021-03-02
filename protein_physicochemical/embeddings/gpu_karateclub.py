import math
import numpy as np
import cupy as cp
import networkx as nx
import cugraph as cnx
from typing import List
# import scipy.stats.mstats
# import scipy.sparse as sparse

import cupyx.scipy.sparse as sparse
from karateclub.estimator import Estimator

class GeoScattering(Estimator):
    r"""An implementation of `"GeoScattering" <http://proceedings.mlr.press/v97/gao19e.html>`_
    from the ICML '19 paper "Geometric Scattering for Graph Data Analysis". The procedure
    uses scattering with wavelet transforms to create graph spectral descriptors. Moments of the
    wavelet transformed features are used as graph level features for the embedding.

    Args:
        order (int): Adjacency matrix powers. Default is 4.
        moments (int): Unnormalized moments considered. Default is 4.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, order: int=4, moments: int=4, seed: int=42):
        self.order = order
        self.moments = moments
        self.seed = seed


    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index = cp.arange(graph.number_of_nodes())
        values = cp.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.coo_matrix((values, (index, index)), shape=shape)
        return D_inverse


    def _get_normalized_adjacency(self, graph):
        """
        Calculating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **A_hat** *(SciPy array)* - The scattering matrix of the graph.
        """
        A = sparse.csr_matrix(nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes())).astype(np.float32))
        D_inverse = self._create_D_inverse(graph)
        A_hat = sparse.identity(graph.number_of_nodes()) + D_inverse.dot(A)
        A_hat = 0.5*A_hat
        return A_hat


    def _calculate_wavelets(self, A_hat):
        """
        Calculating the wavelets of a normalized self-looped adjacency matrix.

        Arg types:
            * **A_hat** *(SciPy array)* - The normalized adjacency matrix.

        Return types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
        """
        Psi = [A_hat.power(2**power) - A_hat.power(2**(power+1)) for power in range(self.order+1)]
        return Psi


    def _create_node_feature_matrix(self, graph):
        """
        Calculating the node features.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph of interest.

        Return types:
            * **X** *(NumPy array)* - The node features.
        """
        log_degree = cp.array([math.log(graph.degree(node)+1) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        eccentricity = cp.array([nx.eccentricity(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        clustering_coefficient = cp.array([nx.clustering(graph, node) for node in range(graph.number_of_nodes())]).reshape(-1, 1)
        X = cp.concatenate([log_degree, eccentricity, clustering_coefficient], axis=1)
        return X


    def _get_zero_order_features(self, X):
        """
        Calculating the zero-th order graph features.

        Arg types:
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The zero-th order graph features.
        """
        features = []
        X = cp.abs(X)
        for col in range(X.shape[1]):
            x = cp.abs(X[:, col])
            for power in range(1, self.order+1):
                features.append(cp.sum(cp.power(x, power)))
        features = cp.array(features).reshape(-1)
        return features


    def _get_first_order_features(self, Psi, X):
        """
        Calculating the first order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The first order graph features.
        """
        features = []
        X = cp.abs(X)
        for col in range(X.shape[1]):
            x = cp.abs(X[:, col])
            for psi in Psi:
                filtered_x = psi.dot(x)
                for q in range(1, self.moments):
                    features.append(cp.sum(cp.power(cp.abs(filtered_x), q)))
        features = cp.array(features).reshape(-1)
        return features


    def _get_second_order_features(self, Psi, X):
        """
        Calculating the second order graph features.

        Arg types:
            * **Psi** *(List of Scipy arrays)* - The wavelet matrices.
            * **X** *(NumPy array)* - The node features.

        Return types:
            * **features** *(NumPy vector)* - The second order graph features.
        """
        features = []
        X = cp.abs(X)
        for col in range(X.shape[1]):
            x = cp.abs(X[:, col])
            for i in range(self.order-1):
                for j in range(i+1, self.order):
                    psi_j = Psi[i]
                    psi_j_prime = Psi[j]     
                    filtered_x = cp.abs(psi_j_prime.dot(cp.abs(psi_j.dot(x))))
                    for q in range(1, self.moments):
                        features.append(cp.sum(cp.power(cp.abs(filtered_x), q)))

        features = cp.array(features).reshape(-1)
        return features


    def _calculate_geoscattering(self, graph):
        """
        Calculating the features of a graph.

        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.

        Return types:
            * **features** *(Numpy vector)* - The embedding of a single graph.
        """
        A_hat = self._get_normalized_adjacency(graph)
        Psi = self._calculate_wavelets(A_hat)
        X = self._create_node_feature_matrix(graph)
        zero_order_features = self._get_zero_order_features(X)
        first_order_features = self._get_first_order_features(Psi, X)
        second_order_features = self._get_second_order_features(Psi, X)
        features = cp.concatenate([zero_order_features, first_order_features, second_order_features], axis=0)
        return features


    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a Geometric-Scattering model.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_geoscattering(graph).asnumpy() for graph in graphs]



    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

class FGSD(Estimator):
    r"""An implementation of `"FGSD" <https://papers.nips.cc/paper/6614-hunt-for-the-unique-stable-sparse-and-fast-feature-learning-on-graphs>`_
    from the NeurIPS '17 paper "Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs".
    The procedure calculates the Moore-Penrose spectrum of the normalized Laplacian.
    Using this spectrum the histogram of the spectral features is used as a whole graph representation. 
    Args:
        hist_bins (int): Number of histogram bins. Default is 200.
        hist_range (int): Histogram range considered. Default is 20.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, hist_bins: int=200, hist_range: int=20, seed: int=42):

        self.hist_bins = hist_bins
        self.hist_range = (0, hist_range)
        self.seed = seed

    def _calculate_fgsd(self, graph):
        """
        Calculating the features of a graph.
        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.
        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        L = sparse.csr_matrix(nx.normalized_laplacian_matrix(graph)).todense()
        fL = cp.linalg.pinv(L)
        ones = cp.ones(L.shape[0])
        S = cp.outer(cp.diag(fL), ones)+cp.outer(ones, cp.diag(fL))-2*fL
        hist, bin_edges = cp.histogram(S.flatten(),
                                       bins=self.hist_bins,
                                       range=self.hist_range)
        return hist

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a FGSD model.
        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_fgsd(graph).asnumpy() for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)


class NetLSD(Estimator):
    r"""An implementation of `"NetLSD" <https://arxiv.org/abs/1805.10712>`_
    from the KDD '18 paper "NetLSD: Hearing the Shape of a Graph". The procedure
    calculate the heat kernel trace of the normalized Laplacian matrix over a
    vector of time scales. If the matrix is large it switches to an approximation
    of the eigenvalues. 
    Args:
        scale_min (float): Time scale interval minimum. Default is -2.0.
        scale_max (float): Time scale interval maximum. Default is 2.0.
        scale_steps (int): Number of steps in time scale. Default is 250.
        scale_approximations (int): Number of eigenvalue approximations. Default is 200.
        seed (int): Random seed value. Default is 42.
    """
    def __init__(self, scale_min: float=-2.0, scale_max: float=2.0,
                 scale_steps: int=250, approximations: int=200, seed: int=42):

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_steps = scale_steps
        self.approximations = approximations
        self.seed = seed
   
    def _calculate_heat_kernel_trace(self, eigenvalues):
        """
        Calculating the heat kernel trace of the normalized Laplacian.
        Arg types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.
        Return types:
            * **heat_kernel_trace** *(Numpy array)* - The heat kernel trace of the graph.
        """
        timescales = cp.logspace(self.scale_min, self.scale_max, self.scale_steps)
        nodes = eigenvalues.shape[0]
        heat_kernel_trace = cp.zeros(timescales.shape)
        for idx, t in enumerate(timescales):
            heat_kernel_trace[idx] = cp.sum(cp.exp(-t * eigenvalues))
        heat_kernel_trace = heat_kernel_trace / nodes
        return heat_kernel_trace

    def _updown_linear_approx(self, eigenvalues_lower, eigenvalues_upper, number_of_nodes):
        """
        Approximating the eigenvalues of the normalized Laplacian.
        Arg types:
            * **eigenvalues_lower** *(Numpy array)* - The smallest eigenvalues of the graph.
            * **eigenvalues_upper** *(Numpy array)* - The largest eigenvalues of the graph.
            * **number_of_nodes** *(int)* - The number of nodes in the graph.
        Return types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.
        """
        nal = len(eigenvalues_lower)
        nau = len(eigenvalues_upper)
        eigenvalues = cp.zeros(number_of_nodes)
        eigenvalues[:nal] = eigenvalues_lower
        eigenvalues[-nau:] = eigenvalues_upper
        eigenvalues[nal-1:-nau+1] = cp.linspace(eigenvalues_lower[-1], eigenvalues_upper[0], number_of_nodes-nal-nau+2)
        return eigenvalues

    def _calculate_eigenvalues(self, laplacian_matrix):
        """
        Calculating the eigenvalues of the normalized Laplacian.
        Arg types:
            * **laplacian_matrix** *(SciPy COO matrix)* - The graph to be decomposed.
        Return types:
            * **eigenvalues** *(Numpy array)* - The eigenvalues of the graph.
        """
        number_of_nodes = laplacian_matrix.shape[0]
        if 2*self.approximations< number_of_nodes:
            lower_eigenvalues = sparse.linalg.eigsh(laplacian_matrix, self.approximations, which="SM", ncv=5*self.approximations, return_eigenvectors=False)[::-1]
            upper_eigenvalues = sparse.linalg.eigsh(laplacian_matrix, self.approximations, which="LM", ncv=5*self.approximations, return_eigenvectors=False)
            eigenvalues = self._updown_linear_approx(lower_eigenvalues, upper_eigenvalues, number_of_nodes)
        else:
            eigenvalues = sparse.linalg.eigsh(laplacian_matrix, number_of_nodes-2, which="LM", return_eigenvectors=False)
        return eigenvalues


    def _calculate_netlsd(self, graph):
        """
        Calculating the features of a graph.
        Arg types:
            * **graph** *(NetworkX graph)* - A graph to be embedded.
        Return types:
            * **hist** *(Numpy array)* - The embedding of a single graph.
        """
        graph.remove_edges_from(nx.selfloop_edges(graph))
        laplacian = sparse.coo_matrix(nx.normalized_laplacian_matrix(graph, nodelist = range(graph.number_of_nodes())), dtype=np.float32)
        eigen_values = self._calculate_eigenvalues(laplacian)
        heat_kernel_trace = self._calculate_heat_kernel_trace(eigen_values)
        return heat_kernel_trace

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a NetLSD model.
        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self._check_graphs(graphs)
        self._embedding = [self._calculate_netlsd(graph).asnumpy() for graph in graphs]


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)