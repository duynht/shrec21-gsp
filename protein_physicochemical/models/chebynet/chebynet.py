import torch
from torch.nn import Linear, LogSoftmax, ReLU, Dropout
from torch_geometric.nn import ChebConv, Set2Set, global_mean_pool
# import torch.nn.functional as F

# class ChebyNet(torch.nn.Module):
#     def __init__(self,
#                  node_input_dim,
#                  output_dim,
#                  node_hidden_dim=64,
#                  polynomial_order=5,
#                  num_step_prop=6,
#                  num_step_set2set=6,
#                  dropout_rate=0.5,
#                  graph_pool='mean'):
#         super(ChebyNet, self).__init__()
#         self.graph_pool = graph_pool.lower()

#         self.num_step_prop = num_step_prop
#         self.lin0 = Linear(node_input_dim, node_hidden_dim)
#         self.conv = ChebConv(node_hidden_dim, node_hidden_dim, K=polynomial_order)

#         if self.graph_pool == 'set2set':
#             self.pool = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
#         elif self.graph_pool == 'mean':
#             self.pool = global_mean_pool(node_hidden_dim)
#         else:
#             raise NotImplementedError("Unknown pooling method")

#         self.lin1 = Linear(2 * node_hidden_dim, node_hidden_dim)
#         self.lin2 = Linear(node_hidden_dim, output_dim)

#         self.dropout = Dropout(p=dropout_rate)
#         self.relu = ReLU()
#         # self.logsm = LogSoftmax(dim=1)

#     def forward(self, data):
#         out = self.lin0(data.x)
#         out = self.relu(out)
        
#         for i in range(self.num_step_prop):            
#             out = self.conv(out, data.edge_index)
#             out = self.relu(out)
#             out = self.dropout(out)

#         out = self.pool(out, data.batch)

#         if self.graph_pool == 'set2set':
#             out = self.lin1(out)
#             out = self.relu(out)

#         out = self.lin2(out)
#         # out = self.logsm(out)
               
#         return out


import torch
from torch.nn import Linear, LogSoftmax, ReLU, Dropout
from torch_cluster import knn_graph
from torch_geometric.nn import ChebConv, Set2Set, global_mean_pool, BatchNorm
# import torch.nn.functional as F

class ChebyNet(torch.nn.Module):
    def __init__(self,
                 node_input_dim,
                 output_dim,
                 node_hidden_dim=64,
                 polynomial_order=5,
                 num_step_prop=6,
                 num_step_set2set=6,
                 dropout_rate=0.5,
                 graph_pool='mean'):
        super(ChebyNet, self).__init__()
        self.graph_pool = graph_pool.lower()

        self.num_step_prop = num_step_prop
        # self.lin0 = Linear(node_input_dim, node_hidden_dim)
        self.conv0 = ChebConv(node_input_dim, node_hidden_dim, K=polynomial_order)
        self.conv1 = ChebConv(node_hidden_dim, node_hidden_dim, K=polynomial_order)
        if self.graph_pool == 'set2set':
            self.pool = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        elif self.graph_pool == 'mean':
            self.pool = global_mean_pool(node_hidden_dim)
        else:
            raise NotImplementedError("Unknown pooling method")

        self.lin1 = Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = Linear(node_hidden_dim, output_dim)

        self.dropout = Dropout(p=dropout_rate)
        self.relu = ReLU()
        # self.logsm = LogSoftmax(dim=1)

        self.bn = BatchNorm(node_hidden_dim)

    def forward(self, data):
        # edge_index = knn_graph(data.pos, k=16, batch=data.batch, loop=True)

        out = self.conv0(data.x, data.edge_index, batch=data.batch)
        out = self.relu(out)
        out = self.bn(out)
        
        for i in range(self.num_step_prop):            
            out = self.conv1(out, data.edge_index, batch=data.batch)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.bn(out)

        out = self.pool(out, batch=data.batch)

        if self.graph_pool == 'set2set':
            out = self.lin1(out)
            out = self.relu(out)
        
        # out = self.bn(out)
        out = self.lin2(out)
        # out = self.logsm(out)
               
        return out