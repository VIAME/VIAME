"""
Functional implementations of common algorithms
"""
import numpy as np
import scipy.optimize


def minimum_weight_assignment(cost):
    """
    Finds optimal assignment between two disjoint sets of items

    Args:
        cost (ndarray): cost[i, j] is the cost between items i and j

    CommandLine:
        xdoctest viame.processes.camtrawl.util_algo minimum_weight_assignment

    Example:
        >>> # Rows are detections in img1, cols are detections in img2
        >>> from viame.processes.camtrawl.util_algo import *
        >>> cost = np.array([
        >>>     [9, 2, 1, 9],
        >>>     [4, 1, 5, 5],
        >>>     [9, 9, 2, 4],
        >>> ])
        >>> assign1 = minimum_weight_assignment(cost)
        >>> print('assign1 = {!r}'.format(assign1))
        assign1 = [(0, 2), (1, 1), (2, 3)]
        >>> assign2 = minimum_weight_assignment(cost.T)
        >>> print('assign2 = {!r}'.format(assign2))
        assign2 = [(1, 1), (2, 0), (3, 2)]
    """
    n1, n2 = cost.shape
    n = max(n1, n2)
    # Embed the [n1 x n2] matrix in a padded (with inf) [n x n] matrix
    cost_matrix = np.full((n, n), fill_value=np.inf)
    cost_matrix[0:n1, 0:n2] = cost

    # Find an effective infinite value for infeasible assignments
    is_infeasible = np.isinf(cost_matrix)
    is_positive = cost_matrix > 0
    feasible_vals = cost_matrix[~(is_infeasible & is_positive)]
    large_val = (n + feasible_vals.sum()) * 2
    # replace infinite values with effective infinite values
    cost_matrix[is_infeasible] = large_val

    # Solve munkres problem for minimum weight assignment
    indexes = list(zip(*scipy.optimize.linear_sum_assignment(cost_matrix)))
    # Return only the feasible assignments
    assignment = [(i, j) for (i, j) in indexes
                  if cost_matrix[i, j] < large_val]
    return assignment
