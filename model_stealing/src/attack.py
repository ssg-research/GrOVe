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

import numpy as np
import torch
import sys
import dgl
import copy
from scipy import sparse

from core.model_handler import ModelHandler
from src.utils import delete_dgl_graph_edge, projection, compute_fidelity
from src.constants import config
from src.train_models import *

def idgl(config):
    model = ModelHandler(config)
    model.train()
    test_metrics, adj = model.test()
    return adj

def estimate_edges(cfg, attack_cfg, train_g):
    if attack_cfg.structure == 'original':
        G_QUERY = train_g
        # Only use node to query
        if attack_cfg.delete_edges == "yes":
            G_QUERY = delete_dgl_graph_edge(train_g)
    
    elif attack_cfg.structure == 'idgl':
        config['dgl_graph'] = train_g
        config['cuda_id'] = cfg.gpu
        adj = idgl(config)
        adj = adj.clone().detach().cpu().numpy()
        if cfg.dataset in ['acm', 'amazon_cs']:
            adj = (adj > 0.9).astype(np.int)
        elif cfg.dataset in ['coauthor_phy']:
            adj = (adj >= 0.999).astype(np.int)
        else:
            adj = (adj > 0.999).astype(np.int)

        sparse_adj = sparse.csr_matrix(adj)
        G_QUERY = dgl.from_scipy(sparse_adj)
        G_QUERY.ndata['features'] = train_g.ndata['features']
        G_QUERY.ndata['labels'] = train_g.ndata['labels']
        G_QUERY = dgl.add_self_loop(G_QUERY)

    else:
        print("Incorrect Structure Parameter. Exiting.")
        sys.exit()

    return G_QUERY

def model_extraction(target_model, train_g, val_g, test_g, device, cfg, log):
    # Set up configuration dictionaries
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    surrogate_cfg = cfg.surrogate_model

    surrogate_projection_accuracy = []
    surrogate_prediction_accuracy = []
    surrogate_embedding_accuracy = []

    surrogate_projection_fidelity = []
    surrogate_prediction_fidelity = []
    surrogate_embedding_fidelity = []

    target_accuracy = []

    # Estimate edges if needed
    G_QUERY = estimate_edges(cfg, attack_cfg, train_g)

    # Query target model with G_QUERY
    if target_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    target_cfg.head, 
                                                                    device)

    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)

    if attack_cfg.structure != 'original':
        print("using idgl reconstructed graph")
        train_g = G_QUERY

    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds
        
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gat_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s, 
                                                                                classifier, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)


    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s, 
                                                                                classifier, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    elif surrogate_cfg.architecture == 'sage':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s, 
                                                                                    classifier, 
                                                                                    test_g, 
                                                                                    test_g.ndata['features'], 
                                                                                    test_g.ndata['labels'], 
                                                                                    test_g.nodes(), 
                                                                                    surrogate_cfg.batch_size, 
                                                                                    device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()

    print("Surrogate Model trained")    

    _acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),test_g.ndata['labels'])
    _predicts = detached_classifier.predict_proba(embds_surrogate.clone().detach().cpu().numpy())

    if target_cfg.architecture == 'sage':
        test_acc, pred, embs = evaluate_sage_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)
        
    elif target_cfg.architecture == 'gat':
        test_acc, pred, embs = evaluate_gat_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, target_cfg.head, device)

    elif target_cfg.architecture == 'gin':
        test_acc, pred, embs = evaluate_gin_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)

    target_accuracy.append(test_acc)

    _fidelity = compute_fidelity(torch.from_numpy(_predicts).to(device), pred.to(device))

    # Which output to save
    if attack_cfg.recovery_from == 'prediction':
        surrogate_prediction_fidelity.append(_fidelity)
        surrogate_prediction_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'embedding':
        surrogate_embedding_fidelity.append(_fidelity)
        surrogate_embedding_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'projection':
        surrogate_projection_fidelity.append(_fidelity)
        surrogate_projection_accuracy.append(_acc)
    else:
        print("wrong recovery-from value")
        sys.exit()

    log.info("########## ATTACK RESULTS ##########")    
    log.info("Target Model Accuracy {}".format(test_acc))
    log.info("Surrogate Accuracy {}".format(_acc))
    log.info("Fidelity {}".format(_fidelity))
    log.info("########## End ATTACK RESULTS ##########")
       
    return model_s, classifier

def double_extraction(target_model, train_g, val_g, test_g, device, cfg, experiment_id, log):
    # Set up configuration dictionaries
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    surrogate_cfg = cfg.surrogate_model

    surrogate_projection_accuracy = []
    surrogate_prediction_accuracy = []
    surrogate_embedding_accuracy = []

    surrogate_projection_fidelity = []
    surrogate_prediction_fidelity = []
    surrogate_embedding_fidelity = []

    target_accuracy = []

    # Split up train dataset into two parts
    train_1_subset, train_2_subset, _ = dgl.data.utils.split_dataset(train_g, frac_list=[0.5, 0.5, 0], shuffle=True, random_state=experiment_id)
    train_1 = train_g.subgraph(train_1_subset.indices)
    train_2 = train_g.subgraph(train_2_subset.indices)

    if not 'features' in train_1.ndata:
        train_1.ndata['features'] = train_1.ndata['feat']
    if not 'labels' in train_1.ndata:
        train_1.ndata['labels'] = train_1.ndata['label']

    if not 'features' in train_2.ndata:
        train_2.ndata['features'] = train_2.ndata['feat']
    if not 'labels' in train_2.ndata:
        train_2.ndata['labels'] = train_2.ndata['label']

    # Estimate edges if needed
    G_QUERY_1 = estimate_edges(cfg, attack_cfg, train_1)
    G_QUERY_2 = estimate_edges(cfg, attack_cfg, train_2)

    if attack_cfg.structure != 'original':
        print("using idgl reconstructed graph")
        train_1 = G_QUERY_1
        train_2 = G_QUERY_2

    train_1.create_formats_()
    train_2.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()


    ## Model Extraction 1
    # Query target model with G_QUERY
    if target_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    target_cfg.head, 
                                                                    device)

    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)


    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, query_preds
        
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY_1.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_gat_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s_1, 
                                                                                classifier_1, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s_1, 
                                                                                classifier_1, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    elif surrogate_cfg.architecture == 'sage':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s_1, 
                                                                                    classifier_1, 
                                                                                    test_g, 
                                                                                    test_g.ndata['features'], 
                                                                                    test_g.ndata['labels'], 
                                                                                    test_g.nodes(), 
                                                                                    surrogate_cfg.batch_size, 
                                                                                    device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()
    
    print("Surrogate 1 trained")

    ## Model Extraction 2
    # Query surrogate model with G_QUERY
    if surrogate_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_surrogate(model_s_1, 
                                                                        classifier_1, 
                                                                        G_QUERY_2, 
                                                                        G_QUERY_2.ndata['features'], 
                                                                        G_QUERY_2.ndata['labels'], 
                                                                        G_QUERY_2.nodes(), 
                                                                        surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_surrogate(model_s_1, 
                                                                        classifier_1, 
                                                                        G_QUERY_2, 
                                                                        G_QUERY_2.ndata['features'], 
                                                                        G_QUERY_2.ndata['labels'], 
                                                                        G_QUERY_2.nodes(), 
                                                                        surrogate_cfg.batch_size, 
                                                                        device)
    elif surrogate_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_surrogate(model_s_1, 
                                                                            classifier_1, 
                                                                            G_QUERY_2, 
                                                                            G_QUERY_2.ndata['features'], 
                                                                            G_QUERY_2.ndata['labels'], 
                                                                            G_QUERY_2.nodes(), 
                                                                            surrogate_cfg.batch_size, 
                                                                            device)
    

    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_2.ndata['features'].shape[1], query_preds.shape[1], train_2, val_g, test_g, query_preds
        
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_2.ndata['features'].shape[1], query_preds.shape[1], train_2, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY_2.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_2.ndata['features'].shape[1], query_preds.shape[1], train_2, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_2, classifier_2, detached_classifier_2 = run_gat_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s_2, 
                                                                                classifier_2, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_2, classifier_2, detached_classifier_2 = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s_2, 
                                                                                classifier_2, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    elif surrogate_cfg.architecture == 'sage':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_2, classifier_2, detached_classifier_2 = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s_2, 
                                                                                    classifier_2, 
                                                                                    test_g, 
                                                                                    test_g.ndata['features'], 
                                                                                    test_g.ndata['labels'], 
                                                                                    test_g.nodes(), 
                                                                                    surrogate_cfg.batch_size, 
                                                                                    device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()
    
    print("Surrogate 2 trained")

    _acc = detached_classifier_2.score(embds_surrogate.clone().detach().cpu().numpy(),test_g.ndata['labels'])
    _predicts = detached_classifier_2.predict_proba(embds_surrogate.clone().detach().cpu().numpy())

    if target_cfg.architecture == 'sage':
        test_acc, pred, embs = evaluate_sage_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)
        
    elif target_cfg.architecture == 'gat':
        test_acc, pred, embs = evaluate_gat_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, target_cfg.head, device)

    elif target_cfg.architecture == 'gin':
        test_acc, pred, embs = evaluate_gin_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)

    target_accuracy.append(test_acc)

    _fidelity = compute_fidelity(torch.from_numpy(_predicts).to(device), pred.to(device))

    # Which output to save
    if attack_cfg.recovery_from == 'prediction':
        surrogate_prediction_fidelity.append(_fidelity)
        surrogate_prediction_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'embedding':
        surrogate_embedding_fidelity.append(_fidelity)
        surrogate_embedding_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'projection':
        surrogate_projection_fidelity.append(_fidelity)
        surrogate_projection_accuracy.append(_acc)
    else:
        print("wrong recovery-from value")
        sys.exit()

    log.info("########## ATTACK RESULTS ##########")    
    log.info("Target Model Accuracy {}".format(test_acc))
    log.info("Surrogate Accuracy {}".format(_acc))
    log.info("Fidelity {}".format(_fidelity))
    log.info("########## End ATTACK RESULTS ##########")
       
    return model_s_2, classifier_2

def fine_tuning(target_model, train_g, val_g, test_g, device, cfg, experiment_id, log):
    # Set up configuration dictionaries
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    surrogate_cfg = cfg.surrogate_model

    surrogate_projection_accuracy = []
    surrogate_prediction_accuracy = []
    surrogate_embedding_accuracy = []

    surrogate_projection_fidelity = []
    surrogate_prediction_fidelity = []
    surrogate_embedding_fidelity = []

    target_accuracy = []

    # Split up train dataset into two parts
    train_1_subset, train_2_subset, _ = dgl.data.utils.split_dataset(train_g, frac_list=[0.5, 0.5, 0], shuffle=True, random_state=experiment_id)
    train_1 = train_g.subgraph(train_1_subset.indices)
    train_2 = train_g.subgraph(train_2_subset.indices)

    if not 'features' in train_1.ndata:
        train_1.ndata['features'] = train_1.ndata['feat']
    if not 'labels' in train_1.ndata:
        train_1.ndata['labels'] = train_1.ndata['label']

    if not 'features' in train_2.ndata:
        train_2.ndata['features'] = train_2.ndata['feat']
    if not 'labels' in train_2.ndata:
        train_2.ndata['labels'] = train_2.ndata['label']

    # Estimate edges if needed
    G_QUERY_1 = estimate_edges(cfg, attack_cfg, train_1)
    G_QUERY_2 = estimate_edges(cfg, attack_cfg, train_2)

    if attack_cfg.structure != 'original':
        print("using idgl reconstructed graph")
        train_1 = G_QUERY_1
        train_2 = G_QUERY_2

    train_1.create_formats_()
    train_2.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()


    ## Model Extraction 
    # Query target model with G_QUERY
    if target_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model, 
                                                                    G_QUERY_1, 
                                                                    G_QUERY_1.ndata['features'], 
                                                                    G_QUERY_1.ndata['labels'], 
                                                                    G_QUERY_1.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    target_cfg.head, 
                                                                    device)

    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)


    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, query_preds
        
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY_1.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_1.ndata['features'].shape[1], query_preds.shape[1], train_1, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_gat_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s_1, 
                                                                                classifier_1, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s_1, 
                                                                                classifier_1, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    elif surrogate_cfg.architecture == 'sage':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_1, classifier_1, detached_classifier_1 = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s_1, 
                                                                                    classifier_1, 
                                                                                    test_g, 
                                                                                    test_g.ndata['features'], 
                                                                                    test_g.ndata['labels'], 
                                                                                    test_g.nodes(), 
                                                                                    surrogate_cfg.batch_size, 
                                                                                    device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()
    
    print("Surrogate trained. Now fine-tuning.")

    ## Fine Tune
    # Preprocess Query Response
    print(cfg.dataset, attack_cfg.recovery_from)
    data = train_2.ndata['features'].shape[1], query_preds.shape[1], train_2, val_g, test_g
        
    # Which Surrogate model to tune
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_2, classifier_2, detached_classifier_2 = tune_gat_surrogate(device, 
                                                                    model_s_1,
                                                                    classifier_1,
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s_2, 
                                                                                classifier_2, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s_2, classifier_2, detached_classifier_2 = tune_gin_surrogate(device,
                                                                    model_s_1,
                                                                    classifier_1, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s_2, 
                                                                                classifier_2, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()
    
    print("Surrogate 2 trained")

    _acc = detached_classifier_2.score(embds_surrogate.clone().detach().cpu().numpy(),test_g.ndata['labels'])
    _predicts = detached_classifier_2.predict_proba(embds_surrogate.clone().detach().cpu().numpy())

    if target_cfg.architecture == 'sage':
        test_acc, pred, embs = evaluate_sage_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)
        
    elif target_cfg.architecture == 'gat':
        test_acc, pred, embs = evaluate_gat_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, target_cfg.head, device)

    elif target_cfg.architecture == 'gin':
        test_acc, pred, embs = evaluate_gin_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)

    target_accuracy.append(test_acc)

    _fidelity = compute_fidelity(torch.from_numpy(_predicts).to(device), pred.to(device))

    # Which output to save
    if attack_cfg.recovery_from == 'prediction':
        surrogate_prediction_fidelity.append(_fidelity)
        surrogate_prediction_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'embedding':
        surrogate_embedding_fidelity.append(_fidelity)
        surrogate_embedding_accuracy.append(_acc)
    elif attack_cfg.recovery_from == 'projection':
        surrogate_projection_fidelity.append(_fidelity)
        surrogate_projection_accuracy.append(_acc)
    else:
        print("wrong recovery-from value")
        sys.exit()

    log.info("########## ATTACK RESULTS ##########")    
    log.info("Target Model Accuracy {}".format(test_acc))
    log.info("Surrogate Accuracy {}".format(_acc))
    log.info("Fidelity {}".format(_fidelity))
    log.info("########## End ATTACK RESULTS ##########")
       
    return model_s_2, classifier_2


def prune_model(model):
    print("Pruning Model")

    pruned_models = {}
    for prune_ratio in np.arange(0.1,0.8,0.1):
        # iterate over all parameters and prune the parameters by multiplying a random bitmask with the parameters
        # random set of bitmask each time so have to run it multiple times
        prune_ratio = round(prune_ratio, 1) # Avoiding floating point arithmetic
        pruned_model = copy.deepcopy(model)
        for _, param in pruned_model.named_parameters():
            bitmask = torch.cuda.FloatTensor(param.shape).uniform_() > prune_ratio
            with torch.no_grad():
                param.copy_(torch.mul(param, bitmask))
        pruned_models[prune_ratio] = pruned_model

    return pruned_models

def model_extraction_with_pruning(target_model, train_g, val_g, test_g, device, cfg, experiment_id, log):
    # Set up configuration dictionaries
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    surrogate_cfg = cfg.surrogate_model

    # Estimate edges if needed
    G_QUERY = estimate_edges(cfg, attack_cfg, train_g)

    torch.manual_seed(experiment_id)

    # Query target model with G_QUERY
    if target_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    target_cfg.head, 
                                                                    device)

    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)

    if attack_cfg.structure != 'original':
        print("using idgl reconstructed graph")
        train_g = G_QUERY

    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds
        
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gat_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        pruned_models = prune_model(model_s)

    elif surrogate_cfg.architecture == 'gin':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        pruned_models = prune_model(model_s)

    elif surrogate_cfg.architecture == 'sage':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        pruned_models = prune_model(model_s)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()

    print("Surrogate Model trained")    
       
    return pruned_models, classifier

def model_extraction_distribution_shift(target_model, train_g, val_g, test_g, device, cfg, experiment_id, log):
    # Set up configuration dictionaries
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    surrogate_cfg = cfg.surrogate_model

    target_accuracy = []

    # Estimate edges if needed
    G_QUERY = estimate_edges(cfg, attack_cfg, train_g)

    # Query target model with G_QUERY
    if target_cfg.architecture == 'sage':
        query_acc, query_preds, query_embs = evaluate_sage_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gin':
        query_acc, query_preds, query_embs = evaluate_gin_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    device)
        
    elif target_cfg.architecture == 'gat':
        query_acc, query_preds, query_embs = evaluate_gat_target(target_model, 
                                                                    G_QUERY, 
                                                                    G_QUERY.ndata['features'], 
                                                                    G_QUERY.ndata['labels'], 
                                                                    G_QUERY.nodes(), 
                                                                    target_cfg.batch_size, 
                                                                    target_cfg.head, 
                                                                    device)

    query_embs = query_embs.to(device)
    query_preds = query_preds.to(device)

    if attack_cfg.structure != 'original':
        print("using idgl reconstructed graph")
        train_g = G_QUERY

    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # Preprocess Query Response
    if attack_cfg.recovery_from == 'prediction':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_preds
    elif attack_cfg.recovery_from == 'embedding':
        print(cfg.dataset, attack_cfg.recovery_from)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, query_embs
        
    elif attack_cfg.recovery_from == 'projection':
        print(cfg.dataset, attack_cfg.recovery_from)
        tsne_embs = projection(query_embs.clone().detach().cpu().numpy(), G_QUERY.ndata['labels'], transform_name=attack_cfg.transform, gnn=target_cfg.architecture, dataset=cfg.dataset)
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(device)
        data = train_g.ndata['features'].shape[1], query_preds.shape[1], train_g, val_g, test_g, tsne_embs
        
    else:
        print("Incorrect Value for recovery-from")
        sys.exit()

    # Which Surrogate model to build
    if surrogate_cfg.architecture == 'gat':
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gat_surrogate_dist_shift(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.head, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(model_s, 
                                                                                classifier, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, surrogate_cfg.head, device)


    elif surrogate_cfg.architecture == 'gin':
        print("NOT CODED YET")
        exit()
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_gin_surrogate(device, 
                                                                    data, 
                                                                    surrogate_cfg.fan_out, 
                                                                    surrogate_cfg.batch_size, 
                                                                    surrogate_cfg.num_workers, 
                                                                    surrogate_cfg.num_hidden, 
                                                                    surrogate_cfg.num_layers, 
                                                                    surrogate_cfg.dropout, 
                                                                    surrogate_cfg.lr, 
                                                                    surrogate_cfg.num_epochs, 
                                                                    surrogate_cfg.log_every, 
                                                                    surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(model_s, 
                                                                                classifier, 
                                                                                test_g, 
                                                                                test_g.ndata['features'], 
                                                                                test_g.ndata['labels'], 
                                                                                test_g.nodes(), 
                                                                                surrogate_cfg.batch_size, 
                                                                                device)

    elif surrogate_cfg.architecture == 'sage':
        print("NOT CODED YET")
        exit()
        print('surrogate model: ', surrogate_cfg.architecture)
        model_s, classifier, detached_classifier = run_sage_surrogate(device, 
                                                                        data, 
                                                                        surrogate_cfg.fan_out, 
                                                                        surrogate_cfg.batch_size, 
                                                                        surrogate_cfg.num_workers, 
                                                                        surrogate_cfg.num_hidden, 
                                                                        surrogate_cfg.num_layers, 
                                                                        surrogate_cfg.dropout, 
                                                                        surrogate_cfg.lr, 
                                                                        surrogate_cfg.num_epochs, 
                                                                        surrogate_cfg.log_every, 
                                                                        surrogate_cfg.eval_every)

        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(model_s, 
                                                                                    classifier, 
                                                                                    test_g, 
                                                                                    test_g.ndata['features'], 
                                                                                    test_g.ndata['labels'], 
                                                                                    test_g.nodes(), 
                                                                                    surrogate_cfg.batch_size, 
                                                                                    device)

    else:
        print("Incorrect value for surrogate-model")
        sys.exit()

    print("Surrogate Model trained")    

    _acc = detached_classifier.score(embds_surrogate.clone().detach().cpu().numpy(),test_g.ndata['labels'])
    _predicts = detached_classifier.predict_proba(embds_surrogate.clone().detach().cpu().numpy())

        
    if target_cfg.architecture == 'gat':
        test_acc, pred, embs = evaluate_gat_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, target_cfg.head, device)

    elif target_cfg.architecture == 'gin':
        test_acc, pred, embs = evaluate_gin_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)

    elif target_cfg.architecture == 'sage':
        test_acc, pred, embs = evaluate_sage_target(target_model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_g.nodes(), target_cfg.batch_size, device)

    target_accuracy.append(test_acc)

    _fidelity = compute_fidelity(torch.from_numpy(_predicts).to(device), pred.to(device))

    log.info("########## ATTACK RESULTS ##########")    
    log.info("Target Model Accuracy {}".format(test_acc))
    log.info("Surrogate Accuracy {}".format(_acc))
    log.info("Fidelity {}".format(_fidelity))
    log.info("########## End ATTACK RESULTS ##########")
       
    return model_s, classifier
    