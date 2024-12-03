from collections.abc import Sequence

import time

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers, tasks, metrics, data
from torchdrug.core import Registry as R

import sys
sys.path.append("..")
from model.architectures import SWE_Pooling

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)
    
@R.register("tasks.FunctionAnnotation")
class FunctionAnnotation(tasks.Task, core.Configurable):

    eps = 1e-10
    _option_members = {"metric"}

    def __init__(self, model, num_class=1, metric=('auprc@micro', 'f1_max'), weight=None, graph_construction_model=None, 
                 mlp_batch_norm=False, mlp_dropout=0, verbose=0, pooling_operation='avg', num_swe_ref_points=1, freeze_swe=False):
        super(FunctionAnnotation, self).__init__()
        self.model = model
        if weight is None:
            weight = torch.ones((num_class,), dtype=torch.float)
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.metric = metric
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        self.mlp = MLP(in_channels=self.model.output_dim,
                       mid_channels=self.model.output_dim,
                       out_channels=num_class,
                       batch_norm=mlp_batch_norm,
                       dropout=mlp_dropout)

        self.pooling_operation = pooling_operation
        self.num_swe_ref_points = num_swe_ref_points
        self.freeze_swe = freeze_swe

        self.swe_pooling = SWE_Pooling(d_in=self.model.output_dim, num_slices=self.model.output_dim, num_ref_points=self.num_swe_ref_points, freeze_swe=self.freeze_swe) # ignored if avg pooling is used

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(self.weight))
        loss = loss_fn(pred.sigmoid(), target)
        
        name = tasks._get_criterion_name("bce")
        metric[name] = loss
        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        graph_feature = self.pooling(graph, output)
        pred = self.mlp(graph_feature)
        return pred

    def pooling(self, graph, output):
        if self.pooling_operation == 'avg':
            graph_feature = output["graph_feature"]
        else:
            concat_residue_feature = output['residue_feature']
            lens = graph.num_residues.to(concat_residue_feature.device)
            padded_residue_feature, mask = layers.functional.variadic_to_padded(concat_residue_feature, lens)
            graph_feature = self.swe_pooling(padded_residue_feature, mask)

        return graph_feature

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric