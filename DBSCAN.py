import numpy as np


def _dist_measurement(point_1, point_2, p=2):
    dist = (np.sum((np.abs(point_1 - point_2))**p))**(1/p)
    return dist

def _get_true_indices(sample):
    indices = set(np.where(sample==1)[0])
    return indices


class DBSCAN:
    def __init__(self,eps=0.25,min_neighbors=15,p=2):
        self.eps = eps
        self.min_neighbors = min_neighbors
        self.p = p

    def _region_query(self, p, X):
        n_samples, n_features = X.shape
        adj = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            dist_i = _dist_measurement(point_1 = p,point_2 = X[i],p = self.p)
            if dist_i <= self.eps:
                adj[i] = 1
        return adj

    def _find_neighbors(self, X):
        n_samples, n_features = X.shape
        neighbors = []
        for i in range(n_samples):
            adj_i = self._region_query(p = X[i], X = X)
            ind_i = _get_true_indices(sample = adj_i)

            neighbors.append(ind_i)

        return neighbors

    def _find_core_points(self, neighbors):
        core_ind = set()
        for i, neighbor_i in enumerate(neighbors):
            if len(neighbor_i) >= self.min_neighbors:
                core_ind.add(i)

        return core_ind

    def _expand_cluster(self, p, neighbors, core_ind, visited, assignment):
        reachable = set(neighbors[p])

        while reachable:
            q = reachable.pop()
            if q not in visited:
                visited.add(q)
                if q in core_ind:
                    reachable |= neighbors[q]
                if q not in assignment:
                    assignment[q] = assignment[p]

    def _assign_to_labels(self, assignment, X):
        n_samples, _ = X.shape
        labels = -1 * np.ones(n_samples, dtype=int)
        for i, cluster_i in assignment.items():
            labels[i] = cluster_i

        return labels

    def fit(self, X):
        X = np.array(X).copy()

        neighbors = self._find_neighbors(X)
        core_ind = self._find_core_points(neighbors)

        assignment = {}
        next_cluster_id = 0
        visited = set()

        for i in core_ind:
            if i not in visited:
                visited.add(i)
                assignment[i] = next_cluster_id
                self._expand_cluster(i, neighbors, core_ind, visited, assignment)
                next_cluster_id += 1

        self.core_sample_indices_ = core_ind
        self.labels_ = self._assign_to_labels(assignment, X)
