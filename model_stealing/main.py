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

import os 
import torch 
import logging
import hydra
from src.train_models import *
from src.data import load_data
from src.attack import model_extraction, double_extraction, model_extraction_with_pruning, model_extraction_distribution_shift, fine_tuning
from src.models import GIN
from src.utils import compute_fidelity, projection
from omegaconf import DictConfig, OmegaConf

MAIN_PATH = '/home/a7waheed/graph-ownership-resolution/model_stealing/'

def load_model(model_cfg, model_architecture, path, g, n_classes, device):
    print(model_architecture)
    if model_architecture == 'gin':

        model = GIN(g.ndata['features'].shape[1],
                        model_cfg.num_hidden,
                        n_classes,
                        model_cfg.num_layers,
                        F.relu,
                        model_cfg.batch_size,
                        model_cfg.num_workers,
                        model_cfg.dropout)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model = torch.load(path)
    
    model = model.to(device)

    return model

def load_surrogate(model_cfg, model_architecture, surrogate_path, classifier_path, g, n_classes, device):
    print(model_architecture)

    classifier = torch.load(classifier_path)

    if model_architecture == 'gin':
        surrogate_model = GINEMB(g.ndata['features'].shape[1],
                        model_cfg.num_hidden,
                        256,
                        n_classes,
                        model_cfg.num_layers,
                        F.relu,
                        model_cfg.batch_size,
                        model_cfg.num_workers,
                        model_cfg.dropout)
        surrogate_model.load_state_dict(torch.load(surrogate_path))
    else:
        surrogate_model = torch.load(surrogate_path)

    surrogate_model = surrogate_model.to(device)

    return surrogate_model, classifier

def train_clean_model(train_g, val_g, test_g, in_feats, labels, g, n_classes, device, model_type, experiment_id, dataset, model_cfg, surrogate_id, surrogate_architecture, surrogate_robustness, attack_structure, recovery_from):
    
    print(model_cfg)

    if model_type == 'independent':
        SAVE_PATH = MAIN_PATH + 'models/%s_%s_%s/' % (model_type, model_cfg.architecture, model_cfg.num_hidden)
        SAVE_NAME = '%s_%s_%s_surrogate_%s_%s_%s_%s_%s_%s' % (
                                                model_type, 
                                                model_cfg.architecture, 
                                                dataset, 
                                                surrogate_architecture, 
                                                surrogate_id, 
                                                surrogate_robustness, 
                                                attack_structure,
                                                recovery_from,
                                                experiment_id)
    else:
        SAVE_PATH = MAIN_PATH + 'models/%s_%s_%s/' % (model_type, model_cfg.architecture, model_cfg.num_hidden)
        SAVE_NAME = '%s_%s_%s_%s' % (model_type, model_cfg.architecture, dataset, experiment_id)
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(SAVE_PATH+SAVE_NAME)
    print(os.path.exists(SAVE_PATH + SAVE_NAME))

    if os.path.exists(SAVE_PATH + SAVE_NAME):
        print("\n MODEL ALREADY TRAINED - LOADING FROM FILE \n")
        trained_model = load_model(model_cfg, model_cfg.architecture, SAVE_PATH + SAVE_NAME, g, n_classes, device)
        return trained_model

    if model_cfg.architecture == "gat":
        data = train_g, val_g, test_g, in_feats, labels, n_classes, g, model_cfg.head
        trained_model = run_gat_target(
            device, data, model_cfg.fan_out, model_cfg.batch_size, model_cfg.num_workers, model_cfg.num_hidden, model_cfg.num_layers, model_cfg.dropout, model_cfg.lr, model_cfg.wd, model_cfg.num_epochs, model_cfg.log_every, model_cfg.eval_every)
        torch.save(trained_model, SAVE_PATH + SAVE_NAME)

    elif model_cfg.architecture == "gin":
        data = train_g, val_g, test_g, in_feats, labels, n_classes
        trained_model = run_gin_target(
            device, data, model_cfg.fan_out, model_cfg.batch_size, model_cfg.num_workers, model_cfg.num_hidden, model_cfg.num_layers, model_cfg.dropout, model_cfg.lr, model_cfg.wd, model_cfg.num_epochs, model_cfg.log_every, model_cfg.eval_every)
        torch.save(trained_model.state_dict(), SAVE_PATH + SAVE_NAME)

    elif model_cfg.architecture == "sage":
        data = in_feats, n_classes, train_g, val_g, test_g
        trained_model = run_sage_target(
            device, data, dataset, model_cfg.fan_out, model_cfg.batch_size, model_cfg.num_workers, model_cfg.num_hidden, model_cfg.num_layers, model_cfg.dropout, model_cfg.lr, model_cfg.wd, model_cfg.num_epochs, model_cfg.log_every, model_cfg.eval_every)
        torch.save(trained_model, SAVE_PATH + SAVE_NAME)

    else:
        raise ValueError("Incorrect model architecture! It should be gat, gin, or sage")

    return trained_model

def evaluate_clean_model(model, val_g, model_cfg, device):
       
    val_nid = val_g.nodes()

    if model_cfg.architecture == "gat":
        val_acc, predictions, embeddings = evaluate_gat_target(model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, model_cfg.batch_size, model_cfg.head, device)

    elif model_cfg.architecture == "gin":
        val_acc, predictions, embeddings = evaluate_gin_target(model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, model_cfg.batch_size, device)

    elif model_cfg.architecture == "sage":
        val_acc, predictions, embeddings = evaluate_sage_target(model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, model_cfg.batch_size, device)   

    return val_acc, predictions, embeddings

def save_results(cfg, target_model, independent_model, surrogate_model, classifier, target_val_g, device, experiment_id, prune_ratio=None):
    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    independent_cfg = cfg.independent_model
    surrogate_cfg = cfg.surrogate_model

    # Save Surrogate Model
    SURROGATE_SAVE_PATH = MAIN_PATH + 'models/surrogate_%s_%s_%s_%s/' % (surrogate_cfg.architecture, surrogate_cfg.num_hidden, attack_cfg.recovery_from, attack_cfg.robustness)
    os.makedirs(SURROGATE_SAVE_PATH, exist_ok=True)
    if prune_ratio == None:
        SURROGATE_SAVE_NAME = SURROGATE_SAVE_PATH + 'surrogate_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset, 
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
    else:
        SURROGATE_SAVE_NAME = SURROGATE_SAVE_PATH + 'surrogate_%s_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset,
                                                                                prune_ratio,  
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
    CLASSIFIER_SAVE_NAME = SURROGATE_SAVE_PATH + 'classifier_%s_%s_%s_%s_target_%s_%s' % (
                                                                            surrogate_cfg.architecture, 
                                                                            cfg.dataset, 
                                                                            attack_cfg.structure,
                                                                            cfg.surrogate_id,
                                                                            target_cfg.architecture,
                                                                            experiment_id)
    if surrogate_cfg.architecture == 'gin':
        torch.save(surrogate_model.state_dict(), SURROGATE_SAVE_NAME)
    else:
        torch.save(surrogate_model, SURROGATE_SAVE_NAME)
    torch.save(classifier, CLASSIFIER_SAVE_NAME)

    # Get embeddings for target and independent models
    val_acc_target, preds_target, embds_target = evaluate_clean_model(target_model, target_val_g, target_cfg, device)
    val_acc_independent, preds_independent, embds_independent = evaluate_clean_model(independent_model, target_val_g, independent_cfg, device)

    # Get embeddings for surrogate model
    if surrogate_cfg.architecture == 'gin':
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gin_surrogate(surrogate_model, classifier, target_val_g, target_val_g.ndata['features'], target_val_g.ndata['labels'], target_val_g.nodes(), surrogate_cfg.batch_size, device)

    elif surrogate_cfg.architecture == 'gat':
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_gat_surrogate(surrogate_model, classifier, target_val_g, target_val_g.ndata['features'], target_val_g.ndata['labels'], target_val_g.nodes(), surrogate_cfg.batch_size, surrogate_cfg.head, device)

    elif surrogate_cfg.architecture == 'sage':
        acc_surrogate, preds_surrogate, embds_surrogate = evaluate_sage_surrogate(surrogate_model, classifier, target_val_g, target_val_g.ndata['features'], target_val_g.ndata['labels'], target_val_g.nodes(), surrogate_cfg.batch_size, device)  

    # Save Results
    RESULTS_PATH = 'results_acc_fidelity_overlap_%s_%s/' % (attack_cfg.overlap, attack_cfg.robustness)
    os.makedirs(MAIN_PATH + RESULTS_PATH, exist_ok=True)
    if prune_ratio == None:
        RESULTS_FILE = MAIN_PATH + RESULTS_PATH + 'target_%s_independent_%s_%s_%s_surrogate_%s_%s_%s_%s_%s.txt' % (target_cfg.architecture, independent_cfg.architecture, cfg.dataset, target_cfg.num_hidden, 
                                                                                    surrogate_cfg.architecture, attack_cfg.recovery_from, attack_cfg.structure, cfg.surrogate_id, experiment_id)
    else:
        RESULTS_FILE = MAIN_PATH + RESULTS_PATH + 'target_%s_independent_%s_%s_%s_surrogate_%s_%s_%s_ratio_%s_%s_%s.txt' % (target_cfg.architecture, independent_cfg.architecture, cfg.dataset, target_cfg.num_hidden, 
                                                                                    surrogate_cfg.architecture, attack_cfg.recovery_from, attack_cfg.structure, prune_ratio, cfg.surrogate_id, experiment_id)

    fidelity = compute_fidelity(preds_target, preds_surrogate)
    with open(RESULTS_FILE, 'w') as f:
        f.write('{},{},{},{}\n'.format(val_acc_target, val_acc_independent, acc_surrogate, fidelity))

    if attack_cfg.recovery_from == 'embedding':
        target_output = embds_target
        independent_output = embds_independent
    elif attack_cfg.recovery_from == 'prediction':
        target_output = preds_target
        independent_output = preds_independent
    elif attack_cfg.recovery_from == 'projection':
        # Target
        target_output = projection(embds_target.clone().detach().cpu().numpy(), target_val_g.ndata['labels'], transform_name=attack_cfg.transform)
        target_output = torch.from_numpy(target_output.values).float().to(device)
        # Independent
        independent_output = projection(embds_independent.clone().detach().cpu().numpy(), target_val_g.ndata['labels'], transform_name=attack_cfg.transform)
        independent_output = torch.from_numpy(independent_output.values).float().to(device)

    surrogate_output = embds_surrogate

    # Save Embeddings
    target_output = target_output.cpu().detach().numpy()
    independent_output = independent_output.cpu().detach().numpy()
    surrogate_output = surrogate_output.cpu().detach().numpy()
    
    if prune_ratio == None:
        EMBEDDINGS_PATH = MAIN_PATH + 'embeddings_overlap_%s_%s_%s_%s/target_%s_independent_%s_%s_%s_surrogate_%s_%s_%s/' % (attack_cfg.overlap, 
                                                                                                                        attack_cfg.recovery_from, 
                                                                                                                        attack_cfg.structure,
                                                                                                                        attack_cfg.robustness,
                                                                                                                        target_cfg.architecture, 
                                                                                                                        independent_cfg.architecture, 
                                                                                                                        cfg.dataset, 
                                                                                                                        target_cfg.num_hidden,
                                                                                                                        surrogate_cfg.architecture, 
                                                                                                                        cfg.surrogate_id,
                                                                                                                        experiment_id)
    else:
        EMBEDDINGS_PATH = MAIN_PATH + 'embeddings_overlap_%s_%s_%s_%s_ratio_%s/target_%s_independent_%s_%s_%s_surrogate_%s_%s_%s/' % (attack_cfg.overlap, 
                                                                                                                        attack_cfg.recovery_from, 
                                                                                                                        attack_cfg.structure,
                                                                                                                        attack_cfg.robustness,
                                                                                                                        prune_ratio,
                                                                                                                        target_cfg.architecture, 
                                                                                                                        independent_cfg.architecture, 
                                                                                                                        cfg.dataset, 
                                                                                                                        target_cfg.num_hidden,
                                                                                                                        surrogate_cfg.architecture, 
                                                                                                                        cfg.surrogate_id,
                                                                                                                        experiment_id)

    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

    with open(EMBEDDINGS_PATH + 'target_embeddings.npy', 'wb') as f:
        np.save(f, target_output)
    with open(EMBEDDINGS_PATH + 'independent_embeddings.npy', 'wb') as f:
        np.save(f, independent_output)
    with open(EMBEDDINGS_PATH + 'surrogate_embeddings.npy', 'wb') as f:
        np.save(f, surrogate_output)

def run_experiment(cfg: DictConfig, log: logging.Logger, experiment_id: int):

    attack_cfg = cfg.attack
    target_cfg = cfg.target_model
    independent_cfg = cfg.independent_model
    surrogate_cfg = cfg.surrogate_model

    print(attack_cfg)
    print(target_cfg)
    print(independent_cfg)
    print(surrogate_cfg)

    if cfg.gpu >= 0:
        device = torch.device('cuda:%d' % cfg.gpu)
    else:
        device = torch.device('cpu')

    # Load data
    target_g, surrogate_g, target_val_g, surrogate_val_g, g, n_classes, in_feats, labels = load_data(cfg.dataset, experiment_id, overlap=attack_cfg.overlap)

    log.info("########## DATA LOADED ##########")
    log.info("# of Target Training Nodes: {}".format(target_g.number_of_nodes()))
    log.info("# of Surrogate Training Nodes: {}".format(surrogate_g.number_of_nodes()))
    log.info("# of Target Val Nodes: {}".format(target_val_g.number_of_nodes()))
    log.info("# of Surrogate Val Nodes: {}".format(surrogate_val_g.number_of_nodes()))
    log.info("################################")

    ### Model Training

    # Train a victim model normally, use surrogate val set as test
    target_model = train_clean_model(target_g, 
                            target_val_g, 
                            surrogate_val_g, 
                            in_feats, 
                            labels, 
                            g, 
                            n_classes, 
                            device, 
                            'target', 
                            experiment_id, 
                            cfg.dataset, 
                            target_cfg, 
                            cfg.surrogate_id,
                            surrogate_cfg.architecture,
                            attack_cfg.robustness,
                            attack_cfg.structure,
                            attack_cfg.recovery_from
                            )


    # Train independent model with the same data, use surrogate val set as test
    independent_model = train_clean_model(target_g, 
                            target_val_g, 
                            surrogate_val_g, 
                            in_feats, 
                            labels, 
                            g, 
                            n_classes, 
                            device, 
                            'independent', 
                            experiment_id, 
                            cfg.dataset, 
                            independent_cfg, 
                            cfg.surrogate_id,
                            surrogate_cfg.architecture,
                            attack_cfg.robustness,
                            attack_cfg.structure,
                            attack_cfg.recovery_from)


    ### Model exctraction

    # Load Surrogate Model if saved
    model_loaded = False
    SURROGATE_SAVE_PATH = MAIN_PATH + 'models/surrogate_%s_%s_%s_%s/' % (surrogate_cfg.architecture, surrogate_cfg.num_hidden, attack_cfg.recovery_from, attack_cfg.robustness)
    if attack_cfg.robustness != 'pruning':
        SURROGATE_SAVE_NAME = SURROGATE_SAVE_PATH + 'surrogate_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset, 
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
        CLASSIFIER_SAVE_NAME = SURROGATE_SAVE_PATH + 'classifier_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset, 
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
        if os.path.exists(SURROGATE_SAVE_NAME):
            print("\n SURROGATE MODEL ALREADY TRAINED - LOADING FROM FILE \n")
            surrogate_model, classifier = load_surrogate(surrogate_cfg, surrogate_cfg.architecture, SURROGATE_SAVE_NAME, CLASSIFIER_SAVE_NAME, g, n_classes, device)
            model_loaded = True
    else:
        SURROGATE_SAVE_NAME = SURROGATE_SAVE_PATH + 'surrogate_%s_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset,
                                                                                0.1,  
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
        CLASSIFIER_SAVE_NAME = SURROGATE_SAVE_PATH + 'classifier_%s_%s_%s_%s_target_%s_%s' % (
                                                                                surrogate_cfg.architecture, 
                                                                                cfg.dataset, 
                                                                                attack_cfg.structure,
                                                                                cfg.surrogate_id,
                                                                                target_cfg.architecture,
                                                                                experiment_id)
        print(SURROGATE_SAVE_NAME)
        if os.path.exists(SURROGATE_SAVE_NAME):
            print("\n SURROGATE MODEL ALREADY TRAINED - LOADING FROM FILE \n")
            pruned_models = {}
            for prune_ratio in np.arange(0.1,0.8,0.1):
                prune_ratio = round(prune_ratio, 1) # Avoiding floating point arithmetic
                SURROGATE_SAVE_NAME = SURROGATE_SAVE_PATH + 'surrogate_%s_%s_%s_%s_%s_target_%s_%s' % (
                                                                                        surrogate_cfg.architecture, 
                                                                                        cfg.dataset,
                                                                                        prune_ratio,  
                                                                                        attack_cfg.structure,
                                                                                        cfg.surrogate_id,
                                                                                        target_cfg.architecture,
                                                                                        experiment_id)
                
                surrogate_model, classifier = load_surrogate(surrogate_cfg, surrogate_cfg.architecture, SURROGATE_SAVE_NAME, CLASSIFIER_SAVE_NAME, g, n_classes, device)
                        
                pruned_models[prune_ratio] = surrogate_model

            model_loaded = True

    if not model_loaded:
        # Train the surrogate model, use target validation set as test
        if attack_cfg.robustness == 'simple_extraction':
            surrogate_model, classifier = model_extraction(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, log)
        elif attack_cfg.robustness == 'double_extraction':
            surrogate_model, classifier = double_extraction(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
        elif attack_cfg.robustness == 'fine_tune':
            surrogate_model, classifier = fine_tuning(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
        elif attack_cfg.robustness == 'pruning':
            pruned_models, classifier = model_extraction_with_pruning(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
        elif attack_cfg.robustness == 'dist_shift':
            surrogate_model, classifier = model_extraction_distribution_shift(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
        else:
            print("Incorrect robustness parameter")
            exit()
    
    # Multiple models with different pruning ratios created
    if attack_cfg.robustness == 'pruning':
        # Get embeddings for surrogate model
        for prune_ratio, surrogate_model in pruned_models.items():
            save_results(cfg, target_model, independent_model, surrogate_model, classifier, target_val_g, device, experiment_id, prune_ratio)
    else:
        save_results(cfg, target_model, independent_model, surrogate_model, classifier, target_val_g, device, experiment_id)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logging.basicConfig(level=logging.INFO, filename='main.log', filemode="w")
    log: logging.Logger = logging.getLogger("All")

    # Run the same experiment multiple times to average results
    run_experiment(cfg, log, cfg.experiment_id)

if __name__ == "__main__":
    main()



