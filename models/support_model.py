import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from models.egnn import EGNN, GNN
from utils.utilis_func import remove_mean, remove_mean_with_mask
import numpy as np
import torch.nn.functional as F


class Node_Predict(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=False,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super().__init__()
        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
            self.in_node_nf = in_node_nf
        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=3 + in_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, x, h, node_mask, edge_mask, context=None):
        bs, n_nodes, n_dims = x.shape
        assert n_dims == self.n_dims
        h_dims = h.shape[-1]
        input_xh = torch.cat([x, h], axis=-1)
        output = self._forward(t, input_xh, node_mask, edge_mask, context)
        output_h = output[:, :, n_dims:].clone()
        output_x = output[:, :, :n_dims].clone()
        return output_x, output_h

    def predict_from_x(self, x, node_mask, edge_mask):
        h_dims = self.in_node_nf
        input_h = torch.ones(
            x.shape[0], x.shape[1], h_dims, dtype=x.dtype, device=x.device
        )
        input_h = input_h / input_h.shape[-1]
        input_h = input_h.to(x.device, x.dtype)
        return self.forward(0, x, input_h, node_mask, edge_mask)

    def loss(self, pred_h, target_h, node_mask):
        bs, n_nodes, h_dims = pred_h.shape
        logits = pred_h.view(bs * n_nodes, -1)
        targets = target_h.view(bs * n_nodes, -1).argmax(dim=-1)
        weight_mask = node_mask.view(bs * n_nodes, -1).squeeze()
        losses = F.cross_entropy(logits, targets, reduction="none")
        losses = losses * weight_mask
        loss = losses.sum() / weight_mask.sum()
        return loss

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        # import pdb
        # pdb.set_trace()
        # print(node_mask)
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == "egnn_dynamics":
            # print(f"{h}\t{x}\t{edges}\t{node_mask}\t{edge_mask}")
            h_final, x_final = self.egnn(
                h, x, edges, node_mask=node_mask, edge_mask=edge_mask
            )
            vel = (
                x_final - x
            ) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print(
                "in node_predict Warning: detected nan, resetting EGNN output to zero."
            )
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def get_assignments(src, dst):
    distance_mtx = cdist(src, dst, metric="euclidean")
    _, dest_ind = linear_sum_assignment(distance_mtx, maximize=False)
    distances = distance_mtx[range(len(dest_ind)), dest_ind]
    return distances, dest_ind


def icp(A, B, max_iterations=100, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        R: final Rotation matrix for A
        rotated: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    src = np.copy(A)
    dst = np.copy(B)

    prev_error = 0

    for i in range(max_iterations):
        # get assignments
        distances, indices = get_assignments(src, dst)

        # compute the transformation between the current source and nearest destination points
        _, R, _ = best_fit_transform(src, dst[indices, :])

        # rotate and update the current source
        src = np.dot(R, src.T).T

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    if i > max_iterations - 1:
        print("out of iteration")

    # calculate final transformation
    _, R, _ = best_fit_transform(A, src)
    A_rotated = np.dot(R, A.T).T
    return R, A_rotated, indices
