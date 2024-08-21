import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Temporal_components import WindowTransformerBlock, STFeatureExtractionLayer
from Graph_learning_representations_module import SpatialCell, BGraphConvolutionalNetwork
from Graph_learning_module import AdaptiveGraphStructureLearningModule


class Regression(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, r_mlp_ratio: float, drop_rate: float):
        super(Regression, self).__init__()
        self.device = torch.device('cuda:0')
        self.flatten_layer = nn.Flatten(start_dim=-2, end_dim=-1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.linear_1 = nn.Linear(in_features=num_nodes * num_features,
                                  out_features=int(r_mlp_ratio * num_nodes * num_features))
        self.linear_2 = nn.Linear(in_features=int(r_mlp_ratio * num_nodes * num_features),
                                  out_features=int((0.5 * r_mlp_ratio) * num_nodes * num_features))
        self.linear_3 = nn.Linear(in_features=int((0.5 * r_mlp_ratio) * num_nodes * num_features),
                                  out_features=1)

    def forward(self, inputs):
        f_out = self.flatten_layer(inputs)
        linear_1 = self.linear_1(f_out)
        linear_1 = self.act(linear_1)
        linear_1 = self.dropout(linear_1)
        linear_2 = self.linear_2(linear_1)
        linear_2 = self.act(linear_2)
        linear_2 = self.dropout(linear_2)
        linear_3 = self.linear_3(linear_2)
        return linear_3


class GatedFusion(nn.Module):
    def __init__(self, num_nodes: int):
        super(GatedFusion, self).__init__()
        self.w_t = nn.Parameter(torch.zeros([num_nodes, num_nodes], requires_grad=True), requires_grad=True)
        self.w_s = nn.Parameter(torch.zeros([num_nodes, num_nodes], requires_grad=True), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros([1, num_nodes], requires_grad=True), requires_grad=True)
        self.act = nn.Sigmoid()
        self.reset_parameters()
        pass

    def reset_parameters(self):
        nn.init.uniform_(self.w_t.data, a=0, b=1)
        nn.init.uniform_(self.w_s.data, a=0, b=1)
        nn.init.uniform_(self.bias.data, a=0, b=1)

    def forward(self, in_t, in_s):
        in_t = torch.matmul(in_t, self.w_t)
        in_s = torch.matmul(in_s.transpose(-1, -2), self.w_s)
        gate = self.act(in_t + in_s + self.bias)
        out = torch.mul(in_t, gate) + torch.mul(in_s, (1.0 - gate))
        return out
        pass


class MSCell(nn.Module):
    def __init__(self, batch: int, num_nodes: int, sequence: int, dim_k: int, num_heads: int, window_size_t: int,
                 shifted_size: int, mlp_ratio: float, drop_rate: float, apt_embed_dim: int):
        super(MSCell, self).__init__()
        self.w_msa = WindowTransformerBlock(num_nodes, dim_k, num_heads, window_size_t,
                                            shifted_size, mlp_ratio, drop_rate, need_mlp=True)
        self.diffusion_conv = SpatialCell(batch, num_nodes, sequence, apt_embed_dim, mlp_ratio,
                                          drop_rate)
        self.gated_f = GatedFusion(num_nodes)
        pass

    def forward(self, inputs, adj_apt):
        out_msa = self.w_msa(inputs)
        out_gcn = self.diffusion_conv(out_msa.transpose(-1, -2), adj_apt)
        out = self.gated_f(out_msa, out_gcn)
        return out
        pass


class MFFusionModule(nn.Module):
    def __init__(self, scales: int, seq: int, dropout: float):
        super(MFFusionModule, self).__init__()
        self.scales = scales
        self.dropout = nn.Dropout(dropout)
        self.weights_list = nn.Parameter(data=torch.empty([scales, seq, seq], requires_grad=True, device='cuda:0'),
                                         requires_grad=True)
        nn.init.xavier_normal_(self.weights_list.data)
        pass

    def forward(self, inputs):
        assert inputs is not None
        out = torch.matmul(self.weights_list[0, :, :].squeeze(0), inputs[0].transpose(-1, -2))
        for i_input in range(1, self.scales):
            out_i = torch.matmul(self.weights_list[i_input, :, :].squeeze(0), inputs[i_input].transpose(-1, -2))
            out += out_i
        # out = self.dropout(out)
        return out


class MultiFusion(nn.Module):
    def __init__(self, seq: int, gain: float):
        super(MultiFusion, self).__init__()
        device = "cuda:0"
        self.gain = gain
        self.weight_s = nn.Parameter(torch.empty([1, 1], requires_grad=True, device=device),
                                     requires_grad=True)
        self.weight_m = nn.Parameter(torch.empty([1, 1], requires_grad=True, device=device),
                                     requires_grad=True)
        self.weight_l = nn.Parameter(torch.empty([1, 1], requires_grad=True, device=device),
                                     requires_grad=True)
        self.weight_l0 = nn.Parameter(torch.empty([1, 1], requires_grad=True, device=device),
                                      requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()
        pass

    def reset_parameters(self):
        nn.init.uniform_(self.weight_s.data, a=0, b=1)
        nn.init.uniform_(self.weight_m.data,  a=0, b=1)
        nn.init.uniform_(self.weight_l.data,  a=0, b=1)
        nn.init.uniform_(self.weight_l0.data,  a=0, b=1)
        pass

    def forward(self, out: list):
        out_0, out_1, out_2, out_3 = out[0], out[1], out[2], out[3]
        weight_con = torch.cat([self.weight_s, self.weight_m, self.weight_l, self.weight_l0], dim=-1)
        weight_con = self.softmax(weight_con)
        weight_s, weight_m, weight_l, weight_l0 = torch.chunk(weight_con, chunks=4, dim=-1)
        out_0 = torch.mul(out_0, weight_s)
        out_1 = torch.mul(out_1, weight_m)
        out_2 = torch.mul(out_2, weight_l)
        out_3 = torch.mul(out_3, weight_l0)
        out = out_0 + out_1 + out_2 + out_3
        return out
        pass


class PMTTransformer(nn.Module):
    def __init__(self, nodes: int, dim_k: int, heads: int,  mlp_ratio: float, dropout_rate: float, window_sizes: list):
        super(PMTTransformer, self).__init__()
        self.tf_modules = nn.ModuleList()
        self.num_layers = len(window_sizes)
        for i, i_size in enumerate(window_sizes):
            i_layer = STFeatureExtractionLayer(num_nodes=nodes, dim_k=dim_k, num_heads=heads,
                                               window_size_t=window_sizes[i], shifted_size=0,
                                               mlp_ratio=mlp_ratio, drop_rate=dropout_rate)
            self.tf_modules.append(i_layer)
        pass

    def forward(self, inputs):
        mappings_list = []
        i_mappings = inputs
        for i_layer, tf_module in enumerate(self.tf_modules):
            i_mappings = tf_module(i_mappings)
            mappings_list.append(i_mappings)
        assert len(mappings_list) == self.num_layers
        return mappings_list


class MBGraphConvolutionalNetwork(nn.Module):
    def __init__(self, scales: int, num_nodes: int, sequence: int, adj_dim: int, dropout: float, gain: float):
        super(MBGraphConvolutionalNetwork, self).__init__()
        self.gsl_modules = nn.ModuleList()
        for i_scale in range(scales):
            gsl_module = AdaptiveGraphStructureLearningModule(num_nodes=num_nodes, sequence=sequence,
                                                              embed_dim=adj_dim, gain=gain, dropout=dropout)
            self.gsl_modules.append(gsl_module)
        self.bgc_networks = nn.ModuleList()
        for i_scale in range(scales):
            bgc_network = BGraphConvolutionalNetwork(sequence=sequence, drop_rate=dropout, gain=gain)
            self.bgc_networks.append(bgc_network)
        self.fusion_module = MFFusionModule(scales, seq=sequence, dropout=dropout)
        pass

    def forward(self, inputs):
        adjs_list = list()
        for i_module, gsl_module in enumerate(self.gsl_modules):
            adj_i = gsl_module()
            adjs_list.append(adj_i)
        mappings_list = list()
        for mappings_i, adj_i, module_i in zip(inputs, adjs_list, self.bgc_networks):
            out_mappings = module_i(mappings_i, adj_i)
            mappings_list.append(out_mappings)
        out = self.fusion_module(mappings_list)
        return out, adjs_list[1]


class RegressionNetwork(nn.Module):
    def __init__(self, nodes: int, seq: int, r_mlp_ratio: float, r_mlp_dropout: float):
        super(RegressionNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.mlp = nn.Sequential(nn.Linear(nodes * seq, int(r_mlp_ratio * nodes * seq)),
                                 nn.ReLU(),
                                 nn.Dropout(r_mlp_dropout),
                                 nn.Linear(int(r_mlp_ratio * nodes * seq), int((0.3 * r_mlp_ratio) * nodes * seq)),
                                 nn.ReLU(),
                                 nn.Dropout(r_mlp_dropout))
        self.regression_layer = nn.Linear(int((0.3 * r_mlp_ratio) * nodes * seq), 1)
        pass

    def forward(self, inputs):
        out = self.flatten(inputs)
        out_mappings = self.mlp(out)
        out = self.regression_layer(out_mappings)
        return out, out_mappings


class MSTSDNModel(nn.Module):
    def __init__(self,  nodes: int, seq: int, dim_k: int, heads: int, window_sizes: list,
                 mlp_ratio: float, dropout_rate: float, adj_dim: int, r_mlp_ratio: float, r_mlp_dropout: float):
        super(MSTSDNModel, self).__init__()
        self.scales = len(window_sizes)
        self.pmtt = PMTTransformer(nodes=nodes, dim_k=dim_k, heads=heads, mlp_ratio=mlp_ratio,
                                   dropout_rate=dropout_rate, window_sizes=window_sizes)
        self.mbgcn = MBGraphConvolutionalNetwork(scales=self.scales, num_nodes=nodes, sequence=seq, adj_dim=adj_dim,
                                                 dropout=dropout_rate)
        self.rn = RegressionNetwork(nodes=nodes, seq=seq, r_mlp_ratio=r_mlp_ratio, r_mlp_dropout=r_mlp_dropout)

    @staticmethod
    def cosine_similarity(static_features):
        source_nodes = []
        values_i = []
        values_all = []
        i = 0
        while True:
            if i > static_features.shape[1] - 1:
                break
            i_series = static_features[:, i:i+1]
            source_nodes.append(i_series)
            i = i + 1
        for i_series in source_nodes:
            for j_series in source_nodes:
                distance = F.cosine_similarity(i_series, j_series, dim=0)
                values_i.append(distance)
            values_all.append(values_i)
            values_i = []
        adj_init = np.asarray(values_all, dtype=np.float)
        adj_init = (adj_init - np.mean(adj_init, keepdims=True)) / np.std(adj_init, keepdims=True)
        adj_init = np.exp(adj_init)
        return adj_init
        pass

    def forward(self, inputs):
        out = self.pmtt(inputs)
        out, adj_one = self.mbgcn(out)
        out, out_mappings = self.rn(out)
        return out, out_mappings, adj_one
        pass


