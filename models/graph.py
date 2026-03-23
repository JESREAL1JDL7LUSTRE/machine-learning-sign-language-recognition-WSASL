import numpy as np

# Source: https://github.com/yysijie/st-gcn
# Added 'mediapipe_51' layout for our skeleton:
#   0       = nose
#   1-2     = shoulders (L, R)
#   3-4     = elbows (L, R)
#   5-6     = wrists (L, R)
#   7-8     = hips (L, R)
#   9-29    = left hand  (wrist root = 9)
#   30-50   = right hand (wrist root = 30)


class Graph():
    """The Graph to model skeletons.

    Args:
        strategy (string): one of 'uniform', 'distance', 'spatial'
        layout (string): one of 'openpose', 'ntu-rgb+d', 'mediapipe_51'
        max_hop (int): maximal distance between two connected nodes
        dilation (int): controls spacing between kernel points
    """

    def __init__(self,
                 layout='mediapipe_51',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop
        )
        self.get_adjacency(strategy)

    def __str__(self):
        return str(self.A)

    def get_edge(self, layout):

        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5),
                (13, 12), (12, 11), (10, 9), (9, 8),
                (11, 5), (8, 2), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
            ]
            self.edge   = self_link + neighbor_link
            self.center = 1

        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            neighbor_link = [(i-1, j-1) for (i, j) in neighbor_1base]
            self.edge   = self_link + neighbor_link
            self.center = 21 - 1

        elif layout == 'mediapipe_51':
            # ── Our filtered MediaPipe skeleton ──────────────────────────────
            # Upper body: 9 joints (0-8)
            #   0=nose, 1=L.shoulder, 2=R.shoulder, 3=L.elbow, 4=R.elbow
            #   5=L.wrist, 6=R.wrist, 7=L.hip, 8=R.hip
            # Left hand: 21 joints (9-29), wrist root = 9
            # Right hand: 21 joints (30-50), wrist root = 30

            self.num_node = 51
            self_link     = [(i, i) for i in range(self.num_node)]

            # Upper body
            upper_body = [
                (0, 1), (0, 2), (1, 2),
                (1, 3), (3, 5),
                (2, 4), (4, 6),
                (1, 7), (2, 8), (7, 8),
            ]

            # Hand connections (generic, applied with offset)
            hand_raw = [
                (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),         # index
                (0, 9), (9, 10), (10, 11), (11, 12),    # middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                (5, 9), (9, 13), (13, 17),               # palm
            ]

            left_hand  = [(a+9,  b+9)  for a, b in hand_raw]
            right_hand = [(a+30, b+30) for a, b in hand_raw]

            # Cross connections: pose wrist → hand wrist root
            cross = [(5, 9), (6, 30)]

            neighbor_link = upper_body + left_hand + right_hand + cross
            self.edge     = self_link + neighbor_link
            self.center   = 0   # nose as center

        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[
                    self.hop_dis == hop
                ]
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
                            if self.hop_dis[j, self.center] == \
                               self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > \
                                 self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


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