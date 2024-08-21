import torch
import torch.nn as nn
from torch.nn import functional as F


def inverse_transformation(adj_ori):
    adj_inv = adj_ori.transpose(-1, -2)
    degree_elements = torch.sum(adj_inv, dim=-1)
    degree = torch.diag(degree_elements)
    degree = torch.linalg.inv(degree)
    adj_inv = torch.matmul(degree, adj_inv)
    return {'original': adj_ori, 'inverse': adj_inv}


class BidirectionalGraphConvolutionalLayer(nn.Module):
    def __init__(self, sequence: int, dropout: float, gain: float):
        super(BidirectionalGraphConvolutionalLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight_f = nn.Linear(in_features=sequence, out_features=sequence, bias=False)
        self.weight_b = nn.Linear(in_features=sequence, out_features=sequence, bias=False)
        self.bias = nn.Parameter(torch.empty([1, sequence], requires_grad=True, device="cuda:0"),
                                 requires_grad=True)
        nn.init.xavier_normal_(self.bias.data, gain=gain)
        self.act = nn.ReLU()

    @staticmethod
    def aggregation_process(inputs, adj):
        if len(adj.shape) == 2:
            out = torch.einsum('nn,bnf->bnf', (adj, inputs))
        else:
            out = torch.bmm(adj, inputs)
        return out

    def forward(self, inputs, adj_apt: dict):
        adj_ori = adj_apt['original']
        adj_inv = adj_apt['inverse']

        ori_out = self.aggregation_process(inputs, adj_ori)
        ori_out = self.weight_f(ori_out)

        inv_out = self.aggregation_process(inputs, adj_inv)
        inv_out = self.weight_b(inv_out)

        out = ori_out + inv_out + self.bias
        out = self.act(out)
        # out = self.dropout(out)
        return out


class BGraphConvolutionalNetwork(nn.Module):
    def __init__(self, sequence: int, drop_rate: float, gain: float):
        super(BGraphConvolutionalNetwork, self).__init__()
        self.bgc_layer = BidirectionalGraphConvolutionalLayer(sequence, dropout=drop_rate, gain=gain)
        self.inv_transform = inverse_transformation
        pass

    def forward(self, inputs, adj_ori):
        res = inputs.transpose(-1, -2)
        adj_apt = self.inv_transform(adj_ori)
        mappings_bgc = self.bgc_layer(inputs.transpose(-1, -2), adj_apt)
        out = res + mappings_bgc
        return out


class MLP(nn.Module):
    def __init__(self, sequence: int, mlp_ratio: float, dropout: float):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(sequence, int(mlp_ratio * sequence))
        self.linear_2 = nn.Linear(int(mlp_ratio * sequence), sequence)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        pass

    def forward(self, inputs):
        out = self.linear_1(inputs)
        out = self.act(out)
        out = self.dropout_1(out)
        out = self.linear_2(out)
        out = self.dropout_2(out)
        return out


class SpatialCell(nn.Module):
    def __init__(self, batch: int, nodes: int, sequence: int, apt_embed_dim: int,
                 mlp_ratio: float, dropout: float, temp: float):
        super(SpatialCell, self).__init__()
        self.diffusion_conv = BGraphConvolutionalNetwork(batch, nodes, sequence, dropout)
        self.proj = nn.Linear(in_features=None, out_features=None)
        pass

    def forward(self, inputs, adj_apt):
        res = inputs
        out = self.diffusion_conv(inputs, adj_apt)
        out = res + out
        return out
        pass


class SpatialAttention(nn.Module):
    def __init__(self, dim_k: int, seq: int, nodes: int, heads: int, window_size: int, dropout: float):
        super(SpatialAttention, self).__init__()
        self.nodes = nodes
        self.heads = heads
        self.head_dim = dim_k // heads
        self.window_size = window_size
        self.softmax = nn.Softmax(dim=-1)
        self.q_proj = nn.Linear(seq, dim_k)
        self.k_proj = nn.Linear(seq, dim_k)
        self.v_proj = nn.Linear(seq, dim_k)
        self.proj = nn.Linear(dim_k, seq)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.func = nn.Conv1d(dim_k, dim_k, kernel_size=3, stride=1, padding='same', groups=dim_k)
        pass

    def window_partition(self, inputs):
        batch, nodes, seq = inputs.shape
        out = inputs.view(batch, nodes // self.window_size, self.window_size, seq)
        out = out.reshape([-1, self.window_size, seq])
        return out

    def window_reverse(self, inputs):
        batch, window_size, seq = inputs.shape
        batch = batch // (self.nodes // window_size)
        out = inputs.view(batch, self.nodes // window_size, window_size, seq)
        out = out.reshape([batch, self.nodes, seq])
        return out

    def get_lepe(self, inputs):
        batch, heads, window_size, head_dim = inputs.shape
        out = inputs.transpose(1, 2).contiguous().view(batch, window_size, heads * head_dim)
        lepe = self.func(out.transpose(-1, -2))
        lepe = lepe.view(batch, heads, head_dim, window_size).transpose(-1, -2)
        out = out.view(batch, window_size, heads, head_dim).transpose(1, 2)
        return out, lepe

    def scale_attention(self, inputs: tuple):
        q, k, v = inputs
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score * (self.head_dim ** -0.5)
        attn = self.softmax(score)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        return out
        pass

    def forward(self, inputs):
        batch, nodes, _ = inputs.shape
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        q = q.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        out = self.scale_attention((q, k, v))
        out = out.transpose(1, 2).contiguous().view(batch, nodes, self.heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
        pass


class GraphAttnLayer(nn.Module):
    def __init__(self, nodes: int, seq: int, dim_k: int, heads: int, dropout: float):
        super(GraphAttnLayer, self).__init__()
        self.heads = heads
        self.head_dim = dim_k // heads
        self.q_proj = nn.Linear(seq, dim_k)
        self.k_proj = nn.Linear(seq, dim_k)
        self.v_proj = nn.Linear(seq, dim_k)
        self.proj = nn.Linear(dim_k, seq)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.attn_ratio = nn.Parameter(torch.empty([1, 1], requires_grad=True, device='cuda:0'),
                                       requires_grad=True)
        nn.init.xavier_normal_(self.attn_ratio.data)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        pass

    def scale_dot_pro_attn(self, inputs: tuple):
        q, k, v = inputs
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score * (self.head_dim ** (-0.5))
        attn = self.softmax(score)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        return out
        pass

    def forward(self, inputs):
        batch, nodes, _ = inputs.shape
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        q = q.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        out = self.scale_dot_pro_attn((q, k, v))
        out = out.transpose(1, 2).contiguous().view(batch, nodes, self.heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
        pass


class SFLayer(nn.Module):
    def __init__(self, dim_k: int, seq: int, nodes: int, heads: int, window_size: int,
                 mlp_ratio: float, dropout: float):
        super(SFLayer, self).__init__()
        self.attention_1 = SpatialAttention(dim_k, seq, nodes, heads, window_size, dropout)
        self.mlp_1 = MLP(seq, mlp_ratio, dropout)
        self.norm_1 = nn.LayerNorm(seq)
        self.norm_2 = nn.LayerNorm(seq)
        self.attention_2 = GraphAttnLayer(nodes, seq, dim_k, heads, dropout)
        self.mlp_2 = MLP(seq, mlp_ratio, dropout)
        self.norm_3 = nn.LayerNorm(seq)
        self.norm_4 = nn.LayerNorm(seq)
        pass

    def forward(self, inputs):
        res = inputs
        out_msa = self.attention_1(inputs)
        out_msa = out_msa + res
        out_norm = self.norm_1(out_msa)
        res = out_norm
        out_mlp = self.mlp_1(out_norm)
        out_mlp = out_mlp + res
        out_norm = self.norm_2(out_mlp)
        res = out_norm
        out_msa = self.attention_2(out_norm)
        out_msa = out_msa + res
        out_norm = self.norm_3(out_msa)
        res = out_norm
        out_mlp = self.mlp_2(out_norm)
        out_mlp = out_mlp + res
        out_norm = self.norm_4(out_mlp)
        return out_norm
        pass
