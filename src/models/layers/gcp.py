import torch
import torch_scatter
from torch_geometric.nn import MessagePassing

def localize(v, frames):
    return torch.bmm(v.unsqueeze(1), frames).squeeze(1)

class GCP(torch.nn.Module):
    def __init__(self, scaler_emb_dim, vector_emb_dim):
        super().__init__()
        self.scalar_emb_dim = scaler_emb_dim
        self.vector_emb_dim = vector_emb_dim
        self.activation = torch.nn.SiLU()

        self.D_s = torch.nn.Sequential(
            torch.nn.Linear(self.vector_emb_dim, 2),
            self.activation
        )
        self.D_z = torch.nn.Sequential(
            torch.nn.Linear(self.vector_emb_dim, self.vector_emb_dim // 4),
            self.activation
        )
        self.U_z = torch.nn.Sequential(
            torch.nn.Linear(self.vector_emb_dim // 4, self.vector_emb_dim),
            self.activation
        )
        self.S_out = torch.nn.Sequential(
            torch.nn.Linear(
                self.scalar_emb_dim + self.vector_emb_dim // 4 + 4,
                self.scalar_emb_dim
            ),
            self.activation
        )
        self.S_out = torch.nn.Sequential(
            torch.nn.Linear(
                self.scalar_emb_dim + self.vector_emb_dim // 4 + 4,
                self.scalar_emb_dim
            ),
            self.activation
        )
        self.V_gate = torch.nn.Sequential(
            torch.nn.Linear(self.scalar_emb_dim, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, s, v, frames):
        v = v.transpose(-1, -2)

        z = self.D_z(v)

        v_s = self.D_s(v)
        v_s = localize(v_s, frames)
        v_s = v_s.view(v_s.shape[0], 4)

        s = torch.cat([
            s,
            self.scalarization(v_s, frames),
            torch.norm(z, dim=-1, keepdim=True)
        ], dim=-1)
        s = self.S_out(s)

        v_up = self.U_z(z)
        v = self.V_gate(s) * v_up

        return s, v


    def scalarization(self, v_s, frames):

        local_scalar_rep_i = torch.matmul(frames, v_s).transpose(-1, -2)

        # reshape frame-derived geometric scalars
        local_scalar_rep_i = local_scalar_rep_i.reshape(v_s.shape[0], 4)

        return local_scalar_rep_i


class GCPMessagePassing(MessagePassing):

    def __init__(self, scaler_emb_dim, vector_emb_dim, **kwargs):
        self.gcp_fusion = GCP(scaler_emb_dim, vector_emb_dim)

    def message_and_aggregate(self, s_i, s_j, v_i, v_j, f_i, f_j, edge_index, **kwargs):
        s_ij = torch.cat([s_i, s_j], dim=-1)
        v_ij = torch.cat([
            localize(v_i, f_i.transpose(-1, -2)),
            localize(v_j, f_j.transpose(-1, -2))
        ], dim=1)
        s_ij, v_ij = self.gcp_fusion(s_ij, v_ij, f_j)
        s_j = torch_scatter.scatter(s_ij, edge_index[0], dim=0, reduce="sum", dim_size=s_j.shape[0])
        v_j = torch_scatter.scatter(v_ij, edge_index[0], dim=0, reduce="sum", dim_size=v_j.shape[0])
        return s_j, v_j

    def forward(self, s, v, frames, edge_index):
        s_update, v_update = self.propagate(edge_index, s=s, v=v, frames=frames)
        return s_update + s, v_update + v

