import torch
import torch.nn as nn


device = torch.device('cuda:0')


class PositionEmbedding(nn.Module):
    def __init__(self, num_nodes: int, sequence: int, dim: int):
        super(PositionEmbedding, self).__init__()
        self.device = torch.device('cuda:0')
        self.sequence_len = sequence
        self.proj_dim_k = nn.Linear(in_features=num_nodes, out_features=dim)
        self.embeddings = nn.Embedding(num_embeddings=sequence, embedding_dim=dim, device=self.device)
        self.drop_0 = nn.Dropout()
        self.drop_1 = nn.Dropout()

    def forward(self, x):
        x = self.proj_dim_k(x)
        x = self.drop_0(x)
        embeddings = self.embeddings(torch.arange(self.sequence_len).to(self.device))
        embeddings = self.drop_1(embeddings)
        return x + embeddings


class MLP(nn.Module):
    def __init__(self, in_dim: int, radio_out: float, dropout_rate: float):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=int(radio_out * in_dim))
        self.linear2 = nn.Linear(in_features=int(radio_out * in_dim), out_features=in_dim)
        self.act_1 = nn.GELU()
        self.act_2 = nn.GELU()
        self.dropout_0 = nn.Dropout(dropout_rate)
        self.dropout_1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_1(x)
        x = self.dropout_0(x)
        x = self.linear2(x)
        x = self.act_2(x)
        x = self.dropout_1(x)
        return x


def chunk_partition(inputs, scale_s, scale_m, scale_l,):
    multiscale_inputs = {}
    small_out = inputs[:, :scale_s, :]
    middle_out = inputs[:, scale_s:(scale_s + scale_m), :]
    large_out = inputs[:, (scale_s + scale_m):(scale_s + scale_m + scale_l), :]
    multiscale_inputs['small_scale'] = small_out
    multiscale_inputs['middle_scale'] = middle_out
    multiscale_inputs['large_scale'] = large_out
    return multiscale_inputs


class MultiscaleSAttn(nn.Module):
    def __init__(self, nodes: int, scale_s: int, scale_m: int, scale_l: int, dropout: float, bias: bool):
        super(MultiscaleSAttn, self).__init__()
        self.nodes = nodes
        self.scale_s = scale_s
        self.scale_m = scale_m
        self.scale_l = scale_l
        self.qkv_s = nn.Linear(in_features=nodes, out_features=3 * nodes, bias=bias)
        self.qkv_m = nn.Linear(in_features=nodes, out_features=3 * nodes, bias=bias)
        self.qkv_l = nn.Linear(in_features=nodes, out_features=3 * nodes, bias=bias)
        self.out_proj = nn.Linear(in_features=nodes, out_features=nodes, bias=bias)
        self.dropout_proj = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scaling = nodes ** (-0.5)
        self.norm = nn.LayerNorm(nodes)
        pass

    def scaled_dot_product_sa(self, q, k):
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn * self.scaling
        attn = self.softmax(attn)
        return attn
        pass

    def forward(self, inputs):
        res = inputs
        inputs = self.norm(inputs)

        multi_scale_inputs = chunk_partition(inputs, self.scale_s, self.scale_m, self.scale_l)
        small_inputs = multi_scale_inputs['small_scale']
        qkv_s = self.qkv_s(small_inputs)
        q_s, k_s, v_s = torch.chunk(qkv_s, chunks=3, dim=-1)
        attn = self.scaled_dot_product_sa(q_s, k_s)
        out_small = torch.matmul(attn, v_s)
        out_small = self.dropout_attn(out_small)

        middle_inputs = multi_scale_inputs['middle_scale']
        qkv_m = self.qkv_m(middle_inputs)
        q_m, k_m, v_m = torch.chunk(qkv_m, chunks=3, dim=-1)
        attn = self.scaled_dot_product_sa(q_m, k_m)
        out_middle = torch.matmul(attn, v_m)
        out_middle = self.dropout_attn(out_middle)

        large_inputs = multi_scale_inputs['large_scale']
        qkv_l = self.qkv_l(large_inputs)
        q_l, k_l, v_l = torch.chunk(qkv_l, chunks=3, dim=-1)
        attn = self.scaled_dot_product_sa(q_l, k_l)
        out_large = torch.matmul(attn, v_l)
        out_large = self.dropout_attn(out_large)

        out = torch.cat([out_small, out_middle, out_large], dim=1)
        out = self.out_proj(out)
        out = self.dropout_proj(out)
        out = out + res
        return out
        pass


class ScaledDotProduct(nn.Module):
    def __init__(self, dim_k: int):
        super(ScaledDotProduct, self).__init__()
        self.scaling = dim_k ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = self.softmax(attn * self.scaling)
        v = torch.matmul(attn, v)
        return v, attn
        pass


class WindowAttention(nn.Module):
    def __init__(self, num_nodes: int, dim_k: int, num_heads: int, window_size: int, dropout: float):
        super(WindowAttention, self).__init__()
        self.dim_k = dim_k
        self.num_heads = num_heads
        self.head_dim = dim_k // num_heads
        self.window_size = window_size
        self.scaling = dim_k ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.q = nn.Linear(in_features=num_nodes, out_features=dim_k, bias=False)
        self.k = nn.Linear(in_features=num_nodes, out_features=dim_k, bias=False)
        self.v = nn.Linear(in_features=num_nodes, out_features=dim_k, bias=False)
        self.proj = nn.Linear(in_features=dim_k, out_features=num_nodes, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        batch, _, _ = q.shape
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = q.reshape([batch, self.window_size, self.num_heads, self.head_dim]).transpose(1, 2)
        k = k.reshape([batch, self.window_size, self.num_heads, self.head_dim]).transpose(1, 2)
        v = v.reshape([batch, self.window_size, self.num_heads, self.head_dim]).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn * self.scaling
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, self.window_size, (self.head_dim * self.num_heads))
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
        pass


def windows_partition(x, window_size):
    batch, seq, nodes = x.shape
    x = x.reshape([batch, seq // window_size, window_size, nodes])
    x = x.reshape([batch * (seq // window_size), window_size, nodes])
    return x
    pass


def windows_reverse(x, seq, window_size):
    batch, _, nodes = x.shape
    batch = batch // (seq // window_size)
    x = x.reshape([batch, seq // window_size, window_size, nodes])
    x = x.reshape([batch, seq, nodes])
    return x
    pass


class WindowTransformerBlock(nn.Module):
    def __init__(self, num_nodes: int, dim_k: int, num_heads: int, window_size_t: int, shifted_size: int,
                 mlp_ratio: float, drop_rate: float, need_mlp: bool):
        super(WindowTransformerBlock, self).__init__()
        self.heads = num_heads
        self.dim_k = dim_k
        self.head_dim = dim_k // num_heads
        self.window_size_t = window_size_t
        self.attn_norm = nn.LayerNorm(num_nodes)
        self.mlp_norm = nn.LayerNorm(num_nodes)
        self.window_attn = WindowAttention(num_nodes, dim_k, num_heads, self.window_size_t, drop_rate)
        if need_mlp:
            self.mlp = MLP(num_nodes, mlp_ratio, drop_rate)
        else:
            self.mlp = None
        self.shifted_size = shifted_size

    def forward(self, x):
        _, seq, _ = x.shape
        res = x
        out_norm = self.attn_norm(x)
        if self.shifted_size > 0:
            shifted_x = torch.roll(out_norm, shifts=self.shifted_size, dims=1)
        else:
            shifted_x = x
        window_x = windows_partition(shifted_x, self.window_size_t)
        out_attn = self.window_attn(window_x, window_x, window_x)
        window_x = windows_reverse(out_attn, seq, self.window_size_t)
        if self.shifted_size > 0:
            shifted_x = torch.roll(window_x, shifts=-self.shifted_size, dims=1)
        else:
            shifted_x = window_x
        x = shifted_x + res
        if self.mlp is not None:
            res = x
            out_norm = self.mlp_norm(x)
            out_mlp = self.mlp(out_norm)
            out = out_mlp + res
            return out
        else:
            return x
        pass


class PatchMerging(nn.Module):
    def __init__(self, nodes: int, dropout: float):
        super(PatchMerging, self).__init__()
        self.reduce_layer = nn.Linear(in_features=2 * nodes, out_features=nodes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(nodes)
        pass

    def forward(self, inputs):
        x_0 = inputs[:, 0::2, :]
        x_1 = inputs[:, 1::2, :]
        out = torch.cat([x_0, x_1], dim=-1)
        out = self.reduce_layer(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out
        pass


class STFeatureExtractionLayer(nn.Module):
    def __init__(self, num_nodes: int, dim_k: int, num_heads: int, window_size_t: int,
                 shifted_size: int, mlp_ratio: float, drop_rate: float):
        super(STFeatureExtractionLayer, self).__init__()
        self.multi_head_wa = WindowTransformerBlock(num_nodes=num_nodes, dim_k=dim_k,
                                                    num_heads=num_heads, window_size_t=window_size_t,
                                                    shifted_size=shifted_size, mlp_ratio=mlp_ratio,
                                                    drop_rate=drop_rate, need_mlp=False)
        shifted_size = window_size_t // 2
        self.multi_head_swa = WindowTransformerBlock(num_nodes=num_nodes, dim_k=dim_k,
                                                     num_heads=num_heads, window_size_t=window_size_t,
                                                     shifted_size=shifted_size, mlp_ratio=mlp_ratio,
                                                     drop_rate=drop_rate, need_mlp=True)

    def forward(self, x):
        out = self.multi_head_wa(x)
        out = self.multi_head_swa(out)
        return out
        pass


class TFAttention(nn.Module):
    def __init__(self, nodes: int, seq: int, k_dim: int, heads: int, window_size: int):
        super(TFAttention, self).__init__()
        self.heads = heads
        self.window_size = window_size
        self.k_dim = k_dim
        self.head_dim = k_dim // heads
        self.q = nn.Linear(in_features=nodes, out_features=k_dim)
        self.k = nn.Linear(in_features=nodes, out_features=k_dim)
        self.v = nn.Linear(in_features=nodes, out_features=k_dim)
        self.proj = nn.Linear(in_features=k_dim, out_features=nodes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_attn = nn.Dropout()
        self.dropout_proj = nn.Dropout()

    def window_partition(self, q):
        batch, seq, k_dim = q.shape
        q = q.reshape([batch, seq // self.window_size, self.window_size, k_dim])
        q = q.view(-1, self.window_size, k_dim)
        return q

    def window_reverse(self, out, seq):
        batch = out.size(0)
        batch = batch // (seq // self.window_size)
        out = out.view(batch, seq // self.window_size, self.window_size, self.k_dim)
        out = out.view(batch, seq, self.k_dim)
        return out
        pass

    def scale_dot_product(self, q, k, v,):
        scaling_factor = q.size(3) ** (-0.5)
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score * scaling_factor
        attn = self.softmax(score)
        attn = self.dropout_attn(attn)
        out = torch.matmul(attn, v)
        return out

    def forward(self, inputs, shifted=None):
        batch, seq, nodes, = inputs.shape
        q, k, v = self.q(inputs), self.k(inputs), self.v(inputs)
        q = self.window_partition(q)
        k = self.window_partition(k)
        v = self.window_partition(v)
        batch = q.size(0)
        q = q.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        out = self.scale_dot_product(q, k, v,)
        out = out.transpose(1, 2).contiguous().view(batch, self.window_size, self.k_dim)
        out = self.window_reverse(out, seq)
        out = self.proj(out)
        out = self.dropout_proj(out)
        return out
        pass


class TFLayer(nn.Module):
    def __init__(self, nodes: int, seq: int, k_dim: int, heads: int,
                 window_size: int, mlp_ratio: float, dropout_rate: float):
        super(TFLayer, self).__init__()
        self.attention_0 = TFAttention(nodes, seq, k_dim, heads, window_size)
        self.attention_1 = TFAttention(nodes, seq, k_dim, heads, window_size)
        self.mlp_0 = MLP(in_dim=nodes, radio_out=mlp_ratio, dropout_rate=dropout_rate)
        self.mlp_1 = MLP(in_dim=nodes, radio_out=mlp_ratio, dropout_rate=dropout_rate)
        self.norm_0 = nn.LayerNorm(nodes)
        self.norm_1 = nn.LayerNorm(nodes)
        self.norm_2 = nn.LayerNorm(nodes)
        self.norm_3 = nn.LayerNorm(nodes)
        pass

    def forward(self, inputs):
        res = inputs
        out_norm = self.norm_0(inputs)
        out_msa = self.attention_0(out_norm)
        out_msa = out_msa + res
        res = out_msa
        out_norm = self.norm_1(out_msa)
        out_mlp = self.mlp_0(out_norm)
        out_mlp = res + out_mlp
        res = out_mlp
        out_norm = self.norm_2(out_mlp)
        out_msa = self.attention_1(out_norm)
        out_msa = out_msa + res
        res = out_msa
        out_norm = self.norm_3(out_msa)
        out_mlp = self.mlp_1(out_norm)
        out_mlp = res + out_mlp
        return out_mlp
        pass





