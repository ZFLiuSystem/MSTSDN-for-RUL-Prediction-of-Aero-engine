import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphStructureLearningModule(nn.Module):
    def __init__(self, num_nodes: int, sequence: int, embed_dim: int, gain: float, k_top: int,
                 need_activation: bool, need_scale: bool, need_adj_return: bool, dropout: float):
        super(AdaptiveGraphStructureLearningModule, self).__init__()
        self.device = torch.device('cuda:0')
        self.num_nodes = num_nodes
        self.sequence = sequence
        self.embed_dim = embed_dim
        self.need_adj_return = need_adj_return
        self.k_top = k_top
        self.senders = nn.Parameter(torch.empty([num_nodes, sequence], requires_grad=True, device="cuda:0"),
                                    requires_grad=True)
        self.receivers = nn.Parameter(torch.empty([num_nodes, sequence], requires_grad=True, device="cuda:0"),
                                      requires_grad=True)
        self.reset_parameters()
        self.scale_m_d = nn.Parameter(torch.empty([1, num_nodes],
                                                  requires_grad=True, dtype=torch.float32).to(self.device),
                                      requires_grad=True).to(self.device)
        nn.init.uniform_(self.scale_m_d.data, a=0, b=1)
        self.scale_m_n = nn.Parameter(torch.empty([1, embed_dim],
                                      requires_grad=True, dtype=torch.float32).to(self.device),
                                      requires_grad=True).to(self.device)
        nn.init.xavier_normal_(self.scale_m_n.data, gain=gain)
        self.linear1 = nn.Linear(sequence, embed_dim, device=self.device)
        self.linear2 = nn.Linear(sequence, embed_dim, device=self.device)
        self.adj_dropout = nn.Dropout(dropout)
        if need_activation:
            self.act_1 = nn.GELU()
            self.act_2 = nn.GELU()
        else:
            self.act = None
        if need_scale:
            self.scale = num_nodes ** -0.5
        else:
            self.scale = None
        pass

    def reset_parameters(self):
        nn.init.xavier_normal_(self.senders.data)
        nn.init.xavier_normal_(self.receivers.data)
        pass

    def top_k_sparse(self, adj):

        mask = torch.zeros([self.num_nodes, self.num_nodes]).to(self.device)
        mask.fill_(0.0)
        source, id_s = torch.topk(adj, k=self.k_top, dim=-1)
        mask.scatter_(1, id_s, 1.0)
        adj = torch.mul(adj, mask) + torch.rand_like(adj) * 0.001
        return adj

    def forward(self):
        senders = self.act_1(self.linear1(self.senders))
        receivers = self.act_2(self.linear2(self.receivers))
        adj1 = torch.matmul(senders, receivers.transpose(-1, -2))
        adj2 = torch.matmul(receivers, senders.transpose(-1, -2))
        adj = adj1 - adj2
        adj_ori = torch.relu(adj)
        adj_ori = adj_ori * self.scale
        adj_ori = F.softmax(adj_ori, dim=-1)
        return adj_ori


class BiTransition(nn.Module):
    def __init__(self):
        super(BiTransition, self).__init__()

    @staticmethod
    def antisymmetric_normal(graph):
        degree = torch.diag(torch.sum(graph, dim=-1) ** (-1))
        return torch.matmul(degree, graph)

    def forward(self, graphs):
        return [
            graphs[0],
            self.antisymmetric_normal(graphs[1])
        ]


class DynamicGraphLearner(nn.Module):
    def __init__(self, batch: int, nodes: int, sequence: int, dropout: float, temp: float):
        super(DynamicGraphLearner, self).__init__()
        self.device = "cuda:0"
        self.q = nn.Linear(in_features=sequence, out_features=int(sequence * 1.5))
        self.k = nn.Linear(in_features=sequence, out_features=int(sequence * 1.5))
        self.act_1 = nn.GELU()
        self.act_2 = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.scaling = (sequence * 1.5) ** (-0.5)
        self.reset_parameters()
        pass

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q.weight, gain=1.44)
        nn.init.xavier_normal_(self.k.weight, gain=1.44)
        pass

    def forward(self, inputs):
        q = self.q(inputs)
        q = self.act_1(q)
        q = self.dropout_1(q)
        k = self.k(inputs)
        k = self.act_2(k)
        k = self.dropout_2(k)
        adj_dy = torch.matmul(q, k.transpose(-1, -2))
        adj_dy = adj_dy * self.scaling
        return adj_dy
        pass


class GlobalAdj(nn.Module):
    def __init__(self, batch: int, num_nodes: int):
        super(GlobalAdj, self).__init__()
        self.device = 'cuda:0'
        self.scale_message = nn.Parameter(torch.zeros([1, num_nodes], requires_grad=True),
                                          requires_grad=True).to(self.device)
        self.weight_dy = nn.Parameter(torch.zeros([num_nodes, num_nodes], requires_grad=True),
                                      requires_grad=True).to(self.device)
        self.weight_apt = nn.Parameter(torch.zeros([num_nodes, num_nodes], requires_grad=True),
                                       requires_grad=True).to(self.device)
        self.act = nn.Softmax(dim=-1)
        self.diag_mark = torch.eye(num_nodes, requires_grad=True,
                                   device=self.device)
        pass

    def forward(self, adj_apt):
        adj_dict = {}
        adj_apt = torch.mul(adj_apt, self.scale_message)
        adj_dict['adj'] = adj_apt
        adj_dict['adj_t'] = adj_apt.transpose(-1, -2)

        out_degree = torch.sum(adj_apt, keepdim=True, dim=1) ** (-1)
        out_degree = torch.mul(self.diag_mark, out_degree)
        adj_dict['out_degree'] = out_degree

        in_degree = torch.sum(adj_apt.transpose(-1, -2), keepdim=True, dim=1) ** (-1)
        in_degree = torch.mul(self.diag_mark, in_degree)
        adj_dict['in_degree'] = in_degree
        return adj_dict
        pass


class UniDireAdj(nn.Module):
    def __init__(self, seq: int, ratio: float):
        super(UniDireAdj, self).__init__()
        self.m_scale = nn.Parameter(torch.empty([1, seq], requires_grad=True, device='cuda:0'),
                                    requires_grad=True)
        self.re = nn.Parameter(torch.empty([1, 1], requires_grad=True, device='cuda:0'),
                               requires_grad=True)
        self.se = nn.Parameter(torch.empty([1, 1], requires_grad=True, device='cuda:0'),
                               requires_grad=True)
        self.reset_param()
        self.proj_1 = nn.Linear(seq, int(seq * ratio))
        self.proj_2 = nn.Linear(seq, int(seq * ratio))
        self.act_1 = nn.GELU()
        self.act_2 = nn.GELU()

    def reset_param(self):
        nn.init.xavier_normal_(self.m_scale.data, gain=1.04)
        nn.init.xavier_normal_(self.re.data, gain=1.04)
        nn.init.xavier_normal_(self.se.data, gain=1.04)

    def forward(self, adj_apt, weight_dif):
        receiver = self.act_1(self.proj_1(adj_apt))
        sender = self.act_2(self.proj_2(adj_apt))
        adj_ori = torch.matmul(sender, receiver.transpose(-1, -2)) - torch.matmul(receiver, sender.transpose(-1, -2))
        adj_ori = F.relu(adj_ori)
        adj_rev = F.relu(torch.mul(adj_ori.transpose(-1, -2), weight_dif))
        return adj_ori, adj_rev


