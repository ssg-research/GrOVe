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
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .utils import compute_acc
from .models import GAT, GIN, SAGE, GATEMB, GINEMB, SAGEEMB, Classification, GATEmbDistShift, Discriminator

import time

MAIN_PATH = '/home/a7waheed/graph-ownership-resolution/model_stealing/'

######## Victim Model Training ########

# GAT target training
def evaluate_gat_target(model, g, inputs, labels, val_nid, batch_size, num_heads, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata['features']
        pred, embds = model.inference(g, inputs, batch_size, num_heads, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), pred, embds


def run_gat_target(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, wd, num_epochs, log_every, eval_every):
    # Unpack data
    train_g, val_g, test_g, in_feats, labels, n_classes, g, num_heads = data

    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model = GAT(in_feats, num_hidden, n_classes, num_layers,
                num_heads, num_workers, F.relu, dropout)
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.to(device) for blk in blocks]

            # Load the input features as well as output labels
            # batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if (epoch + 1) % eval_every == 0 and epoch != 0:
            eval_acc, pred, embds = evaluate_gat_target(
                model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, num_heads, device)
            test_acc, pred, embds = evaluate_gat_target(
                model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, num_heads, device)

            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc {:.4f}'.format(test_acc))


    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    return model

# GIN target training
def evaluate_gin_target(model,
                        g,
                        inputs,
                        labels,
                        val_nid,
                        batch_size,
                        device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        inputs = g.ndata['features']
        pred, embds = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), pred, embds


def run_gin_target(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, wd, num_epochs, log_every, eval_every):
    # Unpack data
    train_g, val_g, test_g, in_feats, labels, n_classes = data

    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model = GIN(in_feats,
                num_hidden,
                n_classes,
                num_layers,
                F.relu,
                batch_size,
                num_workers,
                dropout)
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.to(device) for blk in blocks]

            # Load the input features as well as output labels
            # batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, pred, embds = evaluate_gin_target(
                model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, device)
            test_acc, pred, embds = evaluate_gin_target(
                model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)

            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc {:.4f}'.format(test_acc))
#             if eval_acc > best_eval_acc:
#                 best_eval_acc = eval_acc
#                 best_test_acc = test_acc
#             print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    return model

# SAGE target training
def evaluate_sage_target(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred, embs = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), pred, embs


def run_sage_target(device, data, dataset, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, wd, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data

    # Maintan compability with datasets using train/val/test masks
    if dataset in ['Cora', 'Pubmed', 'Citeseer', 'AIFB', 'Reddit']:
        train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(
            ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    else:
        train_nid = train_g.nodes()
        val_nid = val_g.nodes()
        test_nid = test_g.nodes()

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, num_hidden, n_classes, num_layers,
                 F.relu, batch_size, num_workers, dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd)
    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred, embs = model(blocks, batch_inputs)
            batch_pred = F.softmax(batch_pred, dim=1)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, pred, embs = evaluate_sage_target(
                model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, device)
            test_acc, pred, embs = evaluate_sage_target(
                model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    return model

######## Surrugate Model Training ########

#### Train classifier for surrogate models ####
def train_detached_classifier(test_g, embds_surrogate):

    X = embds_surrogate.clone().detach().cpu().numpy()
    y = test_g.ndata['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=1)

    clf = MLPClassifier(random_state=1, max_iter=300, batch_size=1024)
    
    print("Training detached classifier")
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()

    print("MLP Classifier trained. Time taken: {}".format(end-start))
    clf.predict_proba(X_test[:1])

    clf.predict(X_test[:5, :])

    print("Test Accuracy:", clf.score(X_test, y_test))
    return clf

#### GAT surrogate training ####
def evaluate_gat_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, num_heads, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    # move labels to gpu
    labels = labels.to(device)
    model.eval()
    clf.eval()
    with th.no_grad():
        embs = model.inference(g, inputs, batch_size, num_heads, device)
        embs = embs.to(device)
        logists = clf(embs)

    model.train()
    clf.train()
    return compute_acc(logists, labels), logists, embs


def run_gat_surrogate(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, head, dropout, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, target_response = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = target_response.shape[1]

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = GATEMB(in_feats,
                             num_hidden,
                             n_output_dim,
                             n_classes,
                             num_layers,
                             head,
                             num_workers,
                             F.relu,
                             dropout)

    model_surrogate = model_surrogate.to(device)
    loss_fcn = nn.MSELoss()
    loss_fcn = loss_fcn.to(device)

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    optimizer = optim.Adam(model_surrogate.parameters(), lr=lr)

    clf = Classification(n_output_dim, n_classes)
    clf = clf.to(device)
    optimizer_classification = optim.SGD(clf.parameters(),
                                         lr=0.01)

    print("Starting training loop")

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            loss = th.sqrt(
                loss_fcn(embs, target_response[batch_output_nid]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_classification.zero_grad()
            logists = clf(embs.detach())
            loss_sup = loss_clf(logists, batch_labels)
            loss_sup.backward()
            optimizer_classification.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     val_g,
                                                                     val_g.ndata['features'],
                                                                     val_g.ndata['labels'],
                                                                     val_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     test_g,
                                                                     test_g.ndata['features'],
                                                                     test_g.ndata['labels'],
                                                                     test_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                             clf,
                                                             train_g,
                                                             train_g.ndata['features'],
                                                             train_g.ndata['labels'],
                                                             train_nid,
                                                             batch_size,
                                                             head,
                                                             device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    return model_surrogate, clf, detached_classifier

def tune_gat_surrogate(device, model_surrogate, clf, data, fan_out, batch_size, num_workers, head, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = 256

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = model_surrogate.to(device)
    model_surrogate.train()

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    clf = clf.to(device)
    clf.train()

    params = list(model_surrogate.parameters()) + list(clf.parameters())
    fine_tune_optimizer = optim.Adam(params, lr=lr)
    print("Starting training loop")

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            logists = clf(embs.detach())
            fine_tune_optimizer.zero_grad()
            loss = loss_clf(logists, batch_labels)
            loss.backward()
            fine_tune_optimizer.step()


            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     val_g,
                                                                     val_g.ndata['features'],
                                                                     val_g.ndata['labels'],
                                                                     val_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     test_g,
                                                                     test_g.ndata['features'],
                                                                     test_g.ndata['labels'],
                                                                     test_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                             clf,
                                                             train_g,
                                                             train_g.ndata['features'],
                                                             train_g.ndata['labels'],
                                                             train_nid,
                                                             batch_size,
                                                             head,
                                                             device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    return model_surrogate, clf, detached_classifier

# To do: Change this function to run distirbution shift
def run_gat_surrogate_dist_shift(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, head, dropout, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, target_response = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = target_response.shape[1]

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = GATEmbDistShift(in_feats,
                             num_hidden,
                             n_output_dim,
                             n_classes,
                             num_layers,
                             head,
                             num_workers,
                             F.relu,
                             dropout)

    model_surrogate = model_surrogate.to(device)
    loss_fcn = nn.MSELoss()
    loss_fcn = loss_fcn.to(device)

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    optimizer = optim.Adam(model_surrogate.parameters(), lr=lr)

    clf = Classification(n_output_dim, n_classes)
    clf = clf.to(device)
    optimizer_classification = optim.SGD(clf.parameters(),
                                         lr=0.01)

    discriminator = Discriminator(500, n_output_dim).to(device)
    discriminator = discriminator.to(device)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001)
    optimizer_generator = optim.Adam(model_surrogate.parameters(), lr=0.0001)

    print("Starting training loop")

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            model_surrogate.zero_grad()
            clf.zero_grad()
            discriminator.zero_grad()

            # Compute loss for embedding
            embds = model_surrogate(blocks, batch_inputs)
            loss = th.sqrt(
                loss_fcn(embds, target_response[batch_output_nid]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute loss for classifier
            optimizer_classification.zero_grad()
            logists = clf(embds.detach())
            loss_sup = loss_clf(logists, batch_labels)
            loss_sup.backward()
            optimizer_classification.step()


            # Add for loop, pass them to discriminator to compute loss between guassian and embds
                # embds will have labels, gaussian will have some other labels
                # embeddings will be fake samples, gaussian will be real samples
                # use the log-likelihood loss to update the discriminator
                # compute the loss for the generator to maximize the discriminator loss
                # backpropogate that loss through the initial layers of GNN
            for i in range(0, 5):
                # Backprop on discriminator
                # Tell pytorch that we are running inference only
                model_surrogate.eval()
                gaussian = th.randn_like(embds).to(device)
                real_gauss = discriminator(gaussian)

                embds = model_surrogate(blocks, batch_inputs)
                fake_gauss = discriminator(embds)

                loss_discriminator = -th.mean(th.log(real_gauss) + th.log(1 - fake_gauss))
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Backprop on surrogate (generator)
                model_surrogate.train()
                embds = model_surrogate(blocks, batch_inputs)
                fake_gauss = discriminator(embds)

                loss_generator = -th.mean(th.log(fake_gauss))
                loss_generator.backward()
                optimizer_generator.step()


            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | Disc Loss: {:.4f} | Gen Loss: {:.4f}'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), loss_discriminator.item(), loss_generator.item()))

            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     val_g,
                                                                     val_g.ndata['features'],
                                                                     val_g.ndata['labels'],
                                                                     val_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_gat_surrogate(model_surrogate,
                                                                     clf,
                                                                     test_g,
                                                                     test_g.ndata['features'],
                                                                     test_g.ndata['labels'],
                                                                     test_nid,
                                                                     batch_size,
                                                                     head,
                                                                     device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    eval_acc, eval_preds, eval_embs = evaluate_gat_surrogate(model_surrogate,
                                                             clf,
                                                             train_g,
                                                             train_g.ndata['features'],
                                                             train_g.ndata['labels'],
                                                             train_nid,
                                                             batch_size,
                                                             head,
                                                             device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    return model_surrogate, clf, detached_classifier


#### Train GIN surrogate ####
def evaluate_gin_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    # move labels to gpu
    labels = labels.to(device)
    model.eval()
    clf.eval()
    with th.no_grad():
        embs = model.inference(g, inputs, batch_size, device)
        embs = embs.to(device)
        logists = clf(embs)

    model.train()
    clf.train()
    return compute_acc(logists, labels), logists, embs

def get_intermediate_gin_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    # move labels to gpu
    labels = labels.to(device)
    model.eval()
    clf.eval()
    with th.no_grad():
        embs = model.intermediate_layer_output(g, inputs, batch_size, device)
        embs = embs.to(device)

    return embs

def run_gin_surrogate(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, target_response = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = target_response.shape[1]

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = GINEMB(in_feats,
                             num_hidden,
                             n_output_dim,
                             n_classes,
                             num_layers,
                             F.relu,
                             batch_size,
                             num_workers,
                             dropout)

    model_surrogate = model_surrogate.to(device)
    loss_fcn = nn.MSELoss()
    loss_fcn = loss_fcn.to(device)

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    optimizer = optim.Adam(model_surrogate.parameters(), lr=lr)

    clf = Classification(n_output_dim, n_classes)
    clf = clf.to(device)
    optimizer_classification = optim.SGD(clf.parameters(),
                                         lr=0.01)

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            loss = th.sqrt(
                loss_fcn(embs, target_response[batch_output_nid]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_classification.zero_grad()
            logists = clf(embs.detach())
            loss_sup = loss_clf(logists, batch_labels)
            loss_sup.backward()
            optimizer_classification.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_gin_surrogate(
                model_surrogate, clf, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_gin_surrogate(
                model_surrogate, clf, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    eval_acc, eval_preds, eval_embs = evaluate_gin_surrogate(
        model_surrogate, clf, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, batch_size, device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    return model_surrogate, clf, detached_classifier

def tune_gin_surrogate(device, model_surrogate, clf, data, fan_out, batch_size, num_workers, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = 256

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = model_surrogate.to(device)
    model_surrogate.train()
    
    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    clf = clf.to(device)
    clf.train()

    params = list(model_surrogate.parameters()) + list(clf.parameters())
    fine_tune_optimizer = optim.Adam(params, lr=lr)
    print("Starting training loop")

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            logists = clf(embs.detach())
            fine_tune_optimizer.zero_grad()
            loss = loss_clf(logists, batch_labels)
            loss.backward()
            fine_tune_optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_gin_surrogate(
                model_surrogate, clf, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_gin_surrogate(
                model_surrogate, clf, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    eval_acc, eval_preds, eval_embs = evaluate_gin_surrogate(
        model_surrogate, clf, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, batch_size, device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    return model_surrogate, clf, detached_classifier

#### Train SAGE surrogate ####
def evaluate_sage_surrogate(model, clf, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    # move labels to gpu
    labels = labels.to(device)
    model.eval()
    clf.eval()
    with th.no_grad():
        embs = model.inference(g, inputs, batch_size, device)
        logists = clf(embs)

    model.train()
    clf.train()
    return compute_acc(logists, labels), logists, embs


def run_sage_surrogate(device, data, fan_out, batch_size, num_workers, num_hidden, num_layers, dropout, lr, num_epochs, log_every, eval_every):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g, target_response = data
    train_nid = train_g.nodes()
    val_nid = val_g.nodes()
    test_nid = test_g.nodes()

    n_output_dim = target_response.shape[1]

    print("output dim is: ",  n_output_dim)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model_surrogate = SAGEEMB(in_feats, num_hidden, n_output_dim, n_classes,
                              num_layers, F.relu, batch_size, num_workers, dropout)
    model_surrogate = model_surrogate.to(device)
    loss_fcn = nn.MSELoss()
    loss_fcn = loss_fcn.to(device)

    loss_clf = nn.CrossEntropyLoss()
    loss_clf = loss_clf.to(device)

    optimizer = optim.Adam(model_surrogate.parameters(), lr=lr)

    clf = Classification(n_output_dim, n_classes)
    clf = clf.to(device)
    optimizer_classification = optim.SGD(clf.parameters(),
                                         lr=0.01)

    # Training loop
    avg = 0
    iter_tput = []
    best_val_score = 0.0
    for epoch in range(num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            print("BLOCK SIZE:", len(blocks))
            batch_output_nid = blocks[-1].dstdata['_ID']

            # Compute loss and prediction
            embs = model_surrogate(blocks, batch_inputs)
            loss = th.sqrt(
                loss_fcn(embs, target_response[batch_output_nid]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_classification.zero_grad()
            logists = clf(embs.detach())
            loss_sup = loss_clf(logists, batch_labels)
            loss_sup.backward()
            optimizer_classification.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % log_every == 0:
                acc = compute_acc(logists, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % eval_every == 0 and epoch != 0:
            eval_acc, eval_preds, eval_embs = evaluate_sage_surrogate(
                model_surrogate, clf, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

            test_acc, test_preds, test_embs = evaluate_sage_surrogate(
                model_surrogate, clf, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    eval_acc, eval_preds, eval_embs = evaluate_sage_surrogate(
        model_surrogate, clf, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, batch_size, device)
    detached_classifier = train_detached_classifier(train_g, eval_embs)

    return model_surrogate, clf, detached_classifier
