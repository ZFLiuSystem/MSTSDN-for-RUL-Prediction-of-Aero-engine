import torch
import torch.nn as nn
from GlobalModel import MSTSDNModel
import utils


class Trainer:
    def __init__(self, device: str, lr_rate: float, weight_decay: float, nodes: int,
                 seq: int, dim_k: int, heads: int, window_size: list,
                 mlp_ratio: float, adj_dim: int, r_mlp_ratio: float,
                 dropout_rate: float, r_mlp_dropout: float, milestones: list):
        self.model = MSTSDNModel(nodes=nodes, seq=seq, dim_k=dim_k, heads=heads,
                                 window_sizes=window_size, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate,
                                 adj_dim=adj_dim, r_mlp_ratio=r_mlp_ratio, r_mlp_dropout=r_mlp_dropout)
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                milestones=milestones)
        self.loss = nn.HuberLoss()
        print('\n', self.model)

    def train(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        predictions, _, _ = self.model.forward(inputs)
        loss = self.loss(predictions, labels)
        loss.backward()
        self.optimizer.step()
        root_mse, score = utils.get_metrics(predictions, labels)

        return loss.item(), root_mse, score

    def evaluation(self, inputs, labels):
        self.model.eval()
        predictions, _, _ = self.model(inputs)
        loss = self.loss(predictions, labels)
        root_mse, score = utils.get_metrics(predictions, labels)

        return loss.item(), root_mse, score
