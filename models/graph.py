"""
graph.py — Skeleton Graph Construction and Graph Utilities
==========================================================

This module defines the skeleton graph (joint connectivity) used by ST-GCN
and related models. It supports multiple human body layouts (OpenPose, NTU,
MediaPipe) and several adjacency strategies (uniform, distance, spatial).

Key components:
    Graph              — Builds the adjacency matrix A for a given skeleton layout.
    AdaptiveAdjacency  — Learnable correction on top of the fixed adjacency matrix.
    drop_graph         — DropGraph regularizer (randomly zeroes joints at training time).
    get_hop_distance   — BFS hop-distance matrix between all joint pairs.
    normalize_digraph  — Row-normalize a directed adjacency (D^-1 * A).
    normalize_undigraph— Symmetric normalization (D^-0.5 * A * D^-0.5).

Typical usage:
    graph = Graph(layout='mediapipe_51', strategy='spatial')
    A = torch.tensor(graph.A, dtype=torch.float32)  # shape: (K, V, V)
"""

import numpy as np
import torch
import torch.nn as nn


class Graph():
    """Builds a skeleton graph adjacency matrix for a specified body layout.

    The adjacency matrix A has shape (K, V, V) where:
        K = number of spatial subsets (depends on strategy)
        V = number of joints (nodes)

    Supported layouts:
        'openpose'      — 18-joint OpenPose body layout
        'ntu-rgb+d'     — 25-joint NTU RGB+D body layout
        'mediapipe_51'  — 51-joint MediaPipe layout (body + both hands)

    Supported strategies:
        'uniform'  — all valid hops share a single adjacency matrix (K=1)
        'distance' — one matrix per hop distance (K = max_hop+1)
        'spatial'  — ST-GCN spatial partitioning: root / closer-to-root /
                     further-from-root subsets (K up to 2*max_hop+1)

    Args:
        layout   (str): Name of the skeleton layout. Default: 'mediapipe_51'.
        strategy (str): Partitioning strategy for the adjacency. Default: 'spatial'.
        max_hop  (int): Maximum graph distance (hop) to consider as neighbours.
        dilation (int): Only hops that are multiples of this value are kept.
    """

    def __init__(self, layout='mediapipe_51', strategy='spatial',
                 max_hop=1, dilation=1):
        # Store graph construction parameters
        self.max_hop = max_hop
        self.dilation = dilation

        # Step 1: Build the edge list and set the number of nodes and center joint
        self.get_edge(layout)

        # Step 2: Compute the shortest hop-distance between every pair of joints
        # hop_dis[i, j] = shortest path length between joint i and joint j
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        # Step 3: Build the final adjacency matrix A from the hop distances
        self.get_adjacency(strategy)

    def __str__(self):
        """Pretty-print the adjacency matrix."""
        return str(self.A)

    def get_edge(self, layout):
        """Define joint positions and connections for the chosen layout.

        Sets:
            self.num_node (int): total number of joints in this layout
            self.edge (list): list of (i, j) tuples — all joint connections
                              including self-loops (i, i) and body edges.
            self.center (int): index of the 'root' joint (used by spatial strategy)
        """
        if layout == 'openpose':
            # 18-joint OpenPose body skeleton
            self.num_node = 18
            # Every joint connects to itself (identity / self-loop edges)
            self_link     = [(i, i) for i in range(self.num_node)]
            # Physical body connections (child, parent) pairs
            neighbor_link = [
                (4,3),(3,2),(7,6),(6,5),(13,12),(12,11),(10,9),(9,8),
                (11,5),(8,2),(5,1),(2,1),(0,1),(15,0),(14,0),(17,15),(16,14)
            ]
            self.edge   = self_link + neighbor_link
            self.center = 1   # Neck / mid-torso as root

        elif layout == 'ntu-rgb+d':
            # 25-joint NTU RGB+D skeleton (1-indexed in paper, converted to 0-indexed)
            self.num_node = 25
            self_link     = [(i, i) for i in range(self.num_node)]
            # Edges given in 1-based paper notation then shifted to 0-based
            neighbor_1base = [
                (1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),
                (10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),
                (18,17),(19,18),(20,19),(22,23),(23,8),(24,25),(25,12)
            ]
            neighbor_link = [(i-1, j-1) for (i,j) in neighbor_1base]
            self.edge   = self_link + neighbor_link
            self.center = 20   # Spine base

        elif layout == 'mediapipe_51':
            # 51-joint layout: MediaPipe 9 upper-body joints + 21 left hand + 21 right hand
            # Joint index map:
            #   0       = nose
            #   1,2     = left/right shoulder
            #   3,4     = left/right elbow
            #   5,6     = left/right wrist
            #   7,8     = left/right hip
            #   9–29    = left hand (wrist root at 9)
            #   30–50   = right hand (wrist root at 30)
            self.num_node = 51
            self_link     = [(i, i) for i in range(self.num_node)]

            # Upper body connections (shoulders, elbows, wrists, hips)
            upper_body = [
                (0,1),(0,2),(1,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,8),
            ]

            # Generic hand topology (0 = wrist root, 1–20 = finger joints)
            # Each finger: root→base→mid→tip
            hand_raw = [
                (0,1),(1,2),(2,3),(3,4),          # thumb
                (0,5),(5,6),(6,7),(7,8),           # index
                (0,9),(9,10),(10,11),(11,12),      # middle
                (0,13),(13,14),(14,15),(15,16),    # ring
                (0,17),(17,18),(18,19),(19,20),    # pinky
                (5,9),(9,13),(13,17),              # knuckle row connections
            ]
            # Shift raw hand indices to the left-hand range (joints 9–29)
            left_hand  = [(a+9,  b+9)  for a,b in hand_raw]
            # Shift raw hand indices to the right-hand range (joints 30–50)
            right_hand = [(a+30, b+30) for a,b in hand_raw]
            # Cross-connections: left wrist (joint 5) → left hand root (joint 9)
            # and right wrist (joint 6) → right hand root (joint 30)
            cross      = [(5,9),(6,30)]

            self.edge   = self_link + upper_body + left_hand + right_hand + cross
            self.center = 0   # Nose as the global root joint

        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_adjacency(self, strategy):
        """Build the final adjacency matrix A from hop distances.

        The adjacency is first created as a binary reachability matrix
        (1 if hop distance ≤ max_hop, else 0) and then normalized.

        The strategy then partitions the matrix into K subsets:
            'uniform'  → K=1, one shared matrix
            'distance' → K=max_hop+1, one matrix per exact hop distance
            'spatial'  → K≤2*max_hop+1, root / concentric / centrifugal subsets

        Sets:
            self.A (np.ndarray): shape (K, V, V) — the final adjacency tensor.
        """
        # Only keep hops that are exact multiples of dilation
        valid_hop = range(0, self.max_hop + 1, self.dilation)

        # Build a raw binary adjacency from reachable joints within valid hops
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            # Set edge weight = 1 for all joint pairs at exactly this hop
            adjacency[self.hop_dis == hop] = 1

        # Normalize the full adjacency before partitioning
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            # Single partition: all edges use the same normalized adjacency
            A    = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            # One partition per hop distance
            # A[d][i,j] = weight if hop_dis[i,j] == d, else 0
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A

        elif strategy == 'spatial':
            # ST-GCN spatial partitioning (Yan et al. AAAI 2018):
            # For each hop, edges are split into 3 groups based on relative
            # distance to the root/center joint:
            #   a_root   : same distance from center as target joint
            #   a_close  : source is further from center (centripetal)
            #   a_further: source is closer to center (centrifugal)
            A = []
            for hop in valid_hop:
                a_root    = np.zeros((self.num_node, self.num_node))
                a_close   = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            # Compare both joints' distances to the root
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                # Same depth → root group
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                # j is further from center than i → close group
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                # j is closer to center than i → further group
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    # Self-loops only: one partition (root only)
                    A.append(a_root)
                else:
                    # Non-zero hops: two partitions (close and further)
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE GRAPH TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveAdjacency(nn.Module):
    """Learnable correction added on top of the fixed skeleton adjacency.

    Instead of a purely hand-crafted graph (which assumes a fixed body
    topology), the adaptive adjacency lets the model learn which extra
    joint interactions are helpful for a given task.

    Formulation:
        A_combined = A_fixed + alpha * softmax(A_learnable)

    The learnable matrix is initialized to zeros so that training starts
    from the physics-based graph and gradually introduces data-driven edges.

    Args:
        num_joints  (int)  : Number of body joints / graph nodes (V).
        num_subsets (int)  : Number of spatial subsets (K) from the fixed graph.
        alpha       (float): Scaling coefficient for the learnable term (default 0.1).
    """

    def __init__(self, num_joints, num_subsets, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        # Learnable adjacency parameter: shape (K, V, V)
        # Initialized to near-zero so the physical graph dominates at the start
        self.A_adaptive = nn.Parameter(
            torch.zeros(num_subsets, num_joints, num_joints) * 0.01
        )

    def forward(self, A_fixed):
        """Combine fixed and learnable adjacency matrices.

        Args:
            A_fixed (Tensor): Fixed adjacency from Graph(), shape (K, V, V).

        Returns:
            Tensor: Combined adjacency (K, V, V).
        """
        # Apply softmax over the 'source joint' dimension to keep weights
        # normalised — prevents one joint from dominating all others
        A_learn = torch.softmax(self.A_adaptive, dim=2)
        # Add scaled learned edges on top of the physics-based fixed edges
        return A_fixed + self.alpha * A_learn


# ══════════════════════════════════════════════════════════════════════════════
# DROPGRAPH REGULARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def drop_graph(x, drop_prob=0.1, training=True):
    """DropGraph: randomly zero out entire joint channels during training.

    Analogous to Dropout but operates on the spatial (joint) dimension.
    This forces the model to not over-rely on any single joint, which
    improves robustness to missing or occluded keypoints at inference.

    Reference: Jiang et al. 2021

    Args:
        x        (Tensor): Input feature map, shape (N, C, T, V).
                           N=batch, C=channels, T=frames, V=joints.
        drop_prob (float): Probability of dropping (zeroing) each joint.
                           0.0 disables DropGraph entirely.
        training  (bool) : Only applies during training; identity at eval time.

    Returns:
        Tensor: Feature map with randomly zeroed joints, same shape as x.
    """
    # DropGraph is a no-op at inference time or when disabled
    if not training or drop_prob == 0.0:
        return x

    N, C, T, V = x.shape

    # Build a Bernoulli mask over the joint dimension: (N, 1, 1, V)
    # Each joint independently survives with probability (1 - drop_prob)
    keep_prob = 1.0 - drop_prob
    mask      = torch.bernoulli(
        torch.full((N, 1, 1, V), keep_prob, device=x.device)
    )

    # Scale by 1/keep_prob so the expected output magnitude is unchanged
    # (same as standard inverted dropout scaling)
    return x * mask / keep_prob


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH UTILS
# ══════════════════════════════════════════════════════════════════════════════

def get_hop_distance(num_node, edge, max_hop=1):
    """Compute the shortest hop distance between every pair of joints via BFS.

    The method uses repeated matrix powers of the adjacency matrix:
        If (A^d)[i, j] > 0, joint j is reachable from joint i in exactly d steps.

    Args:
        num_node (int)       : Total number of joints / nodes.
        edge     (list)      : List of (i, j) tuples (undirected edges including self-loops).
        max_hop  (int)       : Maximum hop distance to compute.

    Returns:
        np.ndarray: Shape (num_node, num_node). Entry [i, j] = shortest path
                    length; np.inf if unreachable within max_hop steps.
    """
    # Build a raw symmetric adjacency from the edge list
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Initialise all distances to infinity
    hop_dis      = np.zeros((num_node, num_node)) + np.inf
    # Compute matrix powers A^0, A^1, …, A^max_hop
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # arrive_mat[d][i,j] = True if a path of length d exists from i to j
    arrive_mat   = (np.stack(transfer_mat) > 0)

    # Fill hop_dis in reverse order so shorter paths overwrite longer ones
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    """Row-normalize a directed adjacency matrix: A_norm = A * D^{-1}.

    Each column j is divided by the out-degree of node j, so that messages
    from highly-connected nodes are down-weighted.

    Args:
        A (np.ndarray): Raw adjacency matrix, shape (V, V).

    Returns:
        np.ndarray: Normalized adjacency, shape (V, V).
    """
    # Column-sum = out-degree of each node
    Dl       = np.sum(A, 0)
    num_node = A.shape[0]
    # Build the diagonal inverse-degree matrix D^{-1}
    Dn       = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)   # avoid division by zero
    # A_norm = A @ D^{-1}
    return np.dot(A, Dn)


def normalize_undigraph(A):
    """Symmetric normalization of an undirected adjacency: D^{-1/2} A D^{-1/2}.

    This is the standard GCN normalization (Kipf & Welling 2017).
    It preserves the undirected nature of the graph and provides
    a tighter spectral bound than the directed version.

    Args:
        A (np.ndarray): Symmetric adjacency matrix, shape (V, V).

    Returns:
        np.ndarray: Normalized adjacency, shape (V, V).
    """
    Dl       = np.sum(A, 0)
    num_node = A.shape[0]
    # Build diagonal D^{-1/2} matrix
    Dn       = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    # Symmetric normalization: D^{-1/2} @ A @ D^{-1/2}
    return np.dot(np.dot(Dn, A), Dn)