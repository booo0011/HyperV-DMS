import torch
import torch.nn as nn

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised']

class BatchedHGNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch)

    def forward(self, x, H):
        De = torch.sum(H, dim=1) + 1e-6
        Dv = torch.sum(H, dim=2) + 1e-6
        inv_De = torch.diag_embed(1.0 / De)
        inv_sqrt_Dv = torch.diag_embed(1.0 / torch.sqrt(Dv))
        out = torch.bmm(inv_sqrt_Dv, x)
        out = torch.bmm(H.transpose(1, 2), out)
        out = torch.bmm(inv_De, out)
        out = torch.bmm(H, out)
        out = torch.bmm(inv_sqrt_Dv, out)
        return self.W(out)

class ResumeVoyagerNet(nn.Module):
    def __init__(self, v_dim=956, a_dim=40, hidden_dim=128, num_hyperedges=4):
        super().__init__()
        self.v_proj = nn.Sequential(nn.Linear(v_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.a_proj = nn.Sequential(nn.Linear(a_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.H_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_hyperedges)
        )
        self.batched_hgnn = BatchedHGNNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, 5)

    def forward(self, v_feat, a_feat):
        vh = self.v_proj(v_feat)
        ah = self.a_proj(a_feat)
        nodes = torch.stack([vh, ah], dim=1)
        logits = self.H_generator(nodes)
        H = torch.sigmoid(logits)
        fused_nodes = self.batched_hgnn(nodes, H)
        v_final = vh + fused_nodes[:, 0, :]
        a_final = ah + fused_nodes[:, 1, :]
        final_feat = torch.cat([v_final, a_final], dim=1)
        return self.classifier(final_feat)
