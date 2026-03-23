"""
graph.py
========
Original graph from Yan et al. ST-GCN (AAAI 2018).
Extended with:
  - mediapipe_51 layout
  - Adaptive topology support (learnable adjacency matrix)
  - DropGraph regularization
"""

import numpy as np
import torch
import torch.nn as nn


class Graph():
    """Skeleton graph builder.

    Layouts: 'openpose', 'ntu-rgb+d', 'mediapipe_51'
    Strategies: 'uniform', 'distance', 'spatial'
    """

    def __init__(self, layout='mediapipe_51', strategy='spatial',
                 max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link     = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4,3),(3,2),(7,6),(6,5),(13,12),(12,11),(10,9),(9,8),
                (11,5),(8,2),(5,1),(2,1),(0,1),(15,0),(14,0),(17,15),(16,14)
            ]
            self.edge   = self_link + neighbor_link
            self.center = 1

        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link     = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),
                (10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),
                (18,17),(19,18),(20,19),(22,23),(23,8),(24,25),(25,12)
            ]
            neighbor_link = [(i-1, j-1) for (i,j) in neighbor_1base]
            self.edge   = self_link + neighbor_link
            self.center = 20

        elif layout == 'mediapipe_51':
            self.num_node = 51
            self_link     = [(i, i) for i in range(self.num_node)]

            upper_body = [
                (0,1),(0,2),(1,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,8),
            ]
            hand_raw = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17),
            ]
            left_hand  = [(a+9,  b+9)  for a,b in hand_raw]
            right_hand = [(a+30, b+30) for a,b in hand_raw]
            cross      = [(5,9),(6,30)]

            self.edge   = self_link + upper_body + left_hand + right_hand + cross
            self.center = 0

        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_adjacency(self, strategy):
        valid_hop         = range(0, self.max_hop + 1, self.dilation)
        adjacency         = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A    = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A

        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root    = np.zeros((self.num_node, self.num_node))
                a_close   = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE GRAPH TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveAdjacency(nn.Module):
    """
    Learnable adjacency matrix added on top of the fixed physical graph.

    Based on: Zhang et al. STA-GCN (2020)
    "The model learns data-dependent edge weights rather than relying
    on fixed physical connections."

    A_final = A_physical + alpha * A_learned
    where A_learned is initialized near-zero and learned during training.
    """

    def __init__(self, num_joints, num_subsets, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        # Learnable adjacency: initialized small so physical graph dominates early
        self.A_adaptive = nn.Parameter(
            torch.zeros(num_subsets, num_joints, num_joints) * 0.01
        )

    def forward(self, A_fixed):
        """
        Args:
            A_fixed: (K, V, V) fixed adjacency from Graph()

        Returns:
            (K, V, V) combined adjacency
        """
        # Softmax over source dimension to keep weights normalized
        A_learn = torch.softmax(self.A_adaptive, dim=2)
        return A_fixed + self.alpha * A_learn


# ══════════════════════════════════════════════════════════════════════════════
# DROPGRAPH REGULARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def drop_graph(x, drop_prob=0.1, training=True):
    """
    DropGraph: randomly zero out entire joint channels during training.
    Based on: Jiang et al. "Skeleton Aware Multi-modal SLR" (2021)

    "Selectively dropping noisy or irrelevant joints forces the model
    to learn robust representations that don't over-rely on any single joint."

    Args:
        x         : (N, C, T, V) tensor
        drop_prob : probability of dropping a joint (column in V dim)
        training  : only apply during training

    Returns:
        (N, C, T, V) with some joint channels zeroed
    """
    if not training or drop_prob == 0.0:
        return x

    N, C, T, V = x.shape

    # Create a mask over joints: (N, 1, 1, V)
    keep_prob = 1.0 - drop_prob
    mask      = torch.bernoulli(
        torch.full((N, 1, 1, V), keep_prob, device=x.device)
    )

    # Scale to maintain expected value
    return x * mask / keep_prob


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH UTILS
# ══════════════════════════════════════════════════════════════════════════════

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis      = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat   = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl       = np.sum(A, 0)
    num_node = A.shape[0]
    Dn       = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def normalize_undigraph(A):
    Dl       = np.sum(A, 0)
    num_node = A.shape[0]
    Dn       = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return np.dot(np.dot(Dn, A), Dn)