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

from graphgallery.datasets import NPZDataset
import networkx as nx
import dgl
import numpy as np

def load_graphgallery_data(dataset: str):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset, verbose=False)
    graph = data.graph
    nx_g = nx.from_scipy_sparse_matrix(graph.adj_matrix)

    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = graph.node_attr[node_id].astype(np.float32)
        if dataset in ['blogcatalog', 'flickr']:
            node_data["labels"] = graph.node_label[node_id].astype(np.long) - 1
        else:
            node_data["labels"] = graph.node_label[node_id].astype(np.long)

    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    return dgl_graph, len(np.unique(graph.node_label))

def split_data_overlap(g, experiment_id, surrogate_proportion, frac_list):
    train_subset, val_subset, test_subset = dgl.data.utils.split_dataset(g, frac_list=frac_list, shuffle=True, random_state=experiment_id)
    
    train_g = g.subgraph(train_subset.indices)
    val_g = g.subgraph(val_subset.indices)
    test_g = g.subgraph(test_subset.indices)

    # Surrogate data is a subset of the target model's data
    surrogate_indices = train_subset.indices[:int(len(train_subset.indices) * surrogate_proportion)]
    train_g_surrogate = g.subgraph(surrogate_indices)


    if not 'features' in train_g.ndata:
        train_g.ndata['features'] = train_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        train_g.ndata['labels'] = train_g.ndata['label']

    if not 'features' in val_g.ndata:
        val_g.ndata['features'] = val_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        val_g.ndata['labels'] = val_g.ndata['label']

    if not 'features' in test_g.ndata:
        test_g.ndata['features'] = test_g.ndata['feat']
    if not 'labels' in train_g.ndata:
        test_g.ndata['labels'] = test_g.ndata['label']

    if not 'features' in train_g.ndata:
        train_g_surrogate.ndata['features'] = train_g_surrogate.ndata['feat']
    if not 'labels' in train_g_surrogate.ndata:
        train_g_surrogate.ndata['labels'] = train_g_surrogate.ndata['label']

    train_g.create_formats_()
    train_g_surrogate.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    return train_g, val_g, test_g, train_g_surrogate

def split_data_non_overlap(g, experiment_id, frac_list):
    target_subset, surrogate_subset, val_subset = dgl.data.utils.split_dataset(g, frac_list=frac_list, shuffle=True, random_state=experiment_id)
    
    val_target = val_subset.indices[:int(len(val_subset.indices)*0.5)]
    val_independent = val_subset.indices[int(len(val_subset.indices)*0.5):]

    target_g = g.subgraph(target_subset.indices)
    surrogate_g = g.subgraph(surrogate_subset.indices)
    target_val_g = g.subgraph(val_target)
    surrogate_val_g = g.subgraph(val_independent)

    if not 'features' in target_g.ndata:
        target_g.ndata['features'] = target_g.ndata['feat']
    if not 'labels' in target_g.ndata:
        target_g.ndata['labels'] = target_g.ndata['label']

    if not 'features' in surrogate_g.ndata:
        surrogate_g.ndata['features'] = surrogate_g.ndata['feat']
    if not 'labels' in surrogate_g.ndata:
        surrogate_g.ndata['labels'] = surrogate_g.ndata['label']

    if not 'features' in target_val_g.ndata:
        target_val_g.ndata['features'] = target_val_g.ndata['feat']
    if not 'labels' in target_val_g.ndata:
        target_val_g.ndata['labels'] = target_val_g.ndata['label']

    if not 'features' in surrogate_val_g.ndata:
        surrogate_val_g.ndata['features'] = surrogate_val_g.ndata['feat']
    if not 'labels' in surrogate_val_g.ndata:
        surrogate_val_g.ndata['labels'] = surrogate_val_g.ndata['label']

    target_g.create_formats_()
    surrogate_g.create_formats_()
    target_val_g.create_formats_()
    surrogate_val_g.create_formats_()

    return target_g, surrogate_g, target_val_g, surrogate_val_g


def load_data(dataset: str, experiment_id: int, overlap=False):
    g, n_classes = load_graphgallery_data(dataset)
    in_feats, labels = g.ndata['features'].shape[1], g.ndata['labels']

    if overlap:
        print("NOT CODED YET")
        exit()
        train_g, val_g, test_g, train_g_surrogate = split_data_overlap(g,
                                                                    experiment_id, 
                                                                    surrogate_proportion=0.25, # From victim data 
                                                                    frac_list=[0.45, 0.1, 0.45])
    else:
        target_g, surrogate_g, target_val_g, surrogate_val_g = split_data_non_overlap(g,
                                                                    experiment_id,
                                                                    frac_list=[0.4, 0.4, 0.2])


    return target_g, surrogate_g, target_val_g, surrogate_val_g, g, n_classes, in_feats, labels
