# Authors: Asim Waheed, Vasisht Duddu
# Copyright 2020 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

# GAT victim model
class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_heads,
                 num_workers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.num_workers = num_workers
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads,
                           feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                               num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_classes,
                           num_heads=num_heads, feat_drop=0., attn_drop=0., activation=None, negative_slope=0.2))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h

    def inference(self, g, x, batch_size, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        embeddings = th.zeros(g.number_of_nodes(), self.n_hidden)
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.n_hidden *
                             num_heads if l != len(self.layers) - 1 else self.n_classes)
            else:
                y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(
                    self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                if l < self.n_layers - 1:
                    if l == self.n_layers - 2:
                        h = layer(block, (h, h_dst))
                        emb = h.mean(1)
                        embeddings[output_nodes] = emb.cpu()
                        h = h.flatten(1)
                    else:
                        h = layer(block, (h, h_dst)).flatten(1)
                else:
                    h = layer(block, (h, h_dst))
                    h = h.mean(1)

                y[output_nodes] = h.cpu()

            x = y
        return y, embeddings

# GAT surrogate model
class GATEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 num_heads,
                 num_workers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.num_workers = num_workers
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads,
                           feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                               num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_output_dim,
                           num_heads=num_heads, feat_drop=0., attn_drop=0., activation=None, negative_slope=0.2))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h

    def inference(self, g, x, batch_size, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.n_hidden *
                             num_heads if l != len(self.layers) - 1 else self.n_classes)
            else:
                y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(
                    self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                else:

                    h = layer(block, (h, h_dst))
                    h = h.mean(1)

                y[output_nodes] = h.cpu()

            x = y
        return y

# GAT surrogate with distribution shift
class GATEmbDistShift(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 num_heads,
                 num_workers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.num_workers = num_workers
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads,
                           feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                               num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_output_dim,
                           num_heads=num_heads, feat_drop=0., attn_drop=0., activation=None, negative_slope=0.2))

        # Add extra layer/s for distribution shift
        
    # Will probably remain unchanged? 
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h

    def inference(self, g, x, batch_size, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(g.number_of_nodes(), self.n_hidden *
                             num_heads if l != len(self.layers) - 1 else self.n_classes)
            else:
                y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(
                    self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                if l < self.n_layers - 1:
                    h = layer(block, (h, h_dst)).flatten(1)
                else:

                    h = layer(block, (h, h_dst))
                    h = h.mean(1)

                y[output_nodes] = h.cpu()

            x = y
        return y

# GIN victim model
class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        linear = nn.Linear(in_feats, n_hidden)
        self.layers.append(dglnn.GINConv(linear, 'sum'))
        for i in range(1, n_layers-1):
            linear = nn.Linear(n_hidden, n_hidden)
            self.layers.append(dglnn.GINConv(linear, 'sum'))
        linear = nn.Linear(n_hidden, n_classes)
        self.layers.append(dglnn.GINConv(linear, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_classes)
            if l == len(self.layers) - 2:
                embs = th.zeros(g.number_of_nodes(), self.n_hidden)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l < len(self.layers) - 2:
                    h = self.activation(h)
                    h = self.dropout(h)
                elif l == len(self.layers) - 2:
                    emb = self.activation(h)
                    embs[output_nodes] = emb.cpu()
                    h = self.dropout(emb)

                y[output_nodes] = h.cpu()

            x = y
        return y, embs

# GIN surrogate model
class GINEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        linear = nn.Linear(in_feats, n_hidden)
        self.layers.append(dglnn.GINConv(linear, 'mean'))
        for i in range(1, n_layers-1):
            linear = nn.Linear(n_hidden, n_hidden)
            self.layers.append(dglnn.GINConv(linear, 'mean'))
        linear = nn.Linear(n_hidden, n_output_dim)
        self.layers.append(dglnn.GINConv(linear, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                h = self.activation(h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

    def intermediate_layer_output(self, g, x, batch_size, device):
        # Only propogate till penultimate layer
        for l, layer in enumerate(self.layers):
            if l == len(self.layers)-1:
                continue
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                h = self.activation(h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


# Sage victim model
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'gcn'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'gcn'))

        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x

        for i in range(0, self.n_layers - 1):
            h = self.layers[i](blocks[i], h)
            emb = self.activation(h)
            h = self.dropout(emb)

        h = self.layers[self.n_layers - 1](h)

        return h, emb

    def inference(self, g, x, batch_size, device):
        for l, layer in enumerate(self.layers[:len(self.layers)-1]):

            y = th.zeros(g.number_of_nodes(), self.n_hidden)
            embs = th.zeros(g.number_of_nodes(), self.n_hidden)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                emb = self.activation(h)
                h = self.dropout(emb)

                embs[output_nodes] = emb.cpu()
                y[output_nodes] = h.cpu()

            x = y

        y = self.layers[self.n_layers - 1](x.to(device))
        return y.cpu(), embs

# SAGE surrogate model
class SAGEEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))

        self.layers.append(dglnn.SAGEConv(n_hidden, n_output_dim, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x

        for i in range(0, self.n_layers):
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            if i != self.n_layers - 1:
                h = self.dropout(h)

        return h

    def inference(self, g, x, batch_size, device):

        for l, layer in enumerate(self.layers):

            y = th.zeros(g.number_of_nodes(), self.n_hidden if l !=
                         len(self.layers) - 1 else self.n_output_dim)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                h = self.activation(h)
                if l != self.n_layers - 1:
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()
            x = y

        return y.to(device)

# Classification model for model stealing attacks
class Classification(nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = nn.Linear(emb_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Discriminator model for distribution shift
class Discriminator(nn.Module):  
    def __init__(self,N,z_dim):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return th.sigmoid(self.lin3(x))   