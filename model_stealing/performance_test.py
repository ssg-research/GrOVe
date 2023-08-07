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
from src.attack import model_extraction, double_extraction, model_extraction_with_pruning, model_extraction_distribution_shift
from src.models import GIN
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer

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

def train_clean_model(train_g, val_g, test_g, in_feats, labels, g, n_classes, device, model_type, experiment_id, dataset, model_cfg, surrogate_id, surrogate_architecture, surrogate_robustness, attack_structure, recovery_from):

    if model_type == 'target':
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

    elif model_cfg.architecture == "gin":
        data = train_g, val_g, test_g, in_feats, labels, n_classes
        trained_model = run_gin_target(
            device, data, model_cfg.fan_out, model_cfg.batch_size, model_cfg.num_workers, model_cfg.num_hidden, model_cfg.num_layers, model_cfg.dropout, model_cfg.lr, model_cfg.wd, model_cfg.num_epochs, model_cfg.log_every, model_cfg.eval_every)

    elif model_cfg.architecture == "sage":
        data = in_feats, n_classes, train_g, val_g, test_g
        trained_model = run_sage_target(
            device, data, dataset, model_cfg.fan_out, model_cfg.batch_size, model_cfg.num_workers, model_cfg.num_hidden, model_cfg.num_layers, model_cfg.dropout, model_cfg.lr, model_cfg.wd, model_cfg.num_epochs, model_cfg.log_every, model_cfg.eval_every)

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


    start = timer()
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
    target_train_time = timer() - start

    start = timer()
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
                            attack_cfg.recovery_from
                            )
    independent_train_time = timer() - start

    start = timer()
    ### Model exctraction
    # Train the surrogate model, use target validation set as test
    if attack_cfg.robustness == 'simple_extraction':
        surrogate_model, classifier = model_extraction(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, log)
    elif attack_cfg.robustness == 'double_extraction':
        surrogate_model, classifier = double_extraction(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
    elif attack_cfg.robustness == 'pruning':
        pruned_models, classifier = model_extraction_with_pruning(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
    elif attack_cfg.robustness == 'dist_shift':
        surrogate_model, classifier = model_extraction_distribution_shift(target_model, surrogate_g, surrogate_val_g, target_val_g, device, cfg, experiment_id, log)
    else:
        print("Incorrect robustness parameter")
        exit()
    surrogate_train_time = timer() - start

    time_filename = MAIN_PATH + 'times.csv'
    if os.path.exists(time_filename):
        time_file = open(time_filename, 'a')
    else:
        time_file = open(time_filename, 'w')
        time_file.write('dataset,target_architecture,independent_architecture,surrogate_architecture,attack_type,robustness,experiment_id,independent_time,surrogate_time\n')

    time_file.write('{},{},{},{},{},{},{},{},{}\n'.format(
        cfg.dataset,
        target_cfg.architecture,
        independent_cfg.architecture,
        surrogate_cfg.architecture,
        attack_cfg.structure,
        attack_cfg.robustness,
        experiment_id,
        independent_train_time,
        surrogate_train_time
    ))

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logging.basicConfig(level=logging.INFO, filename='main.log', filemode="w")
    log: logging.Logger = logging.getLogger("All")

    # Run the same experiment multiple times to average results
    run_experiment(cfg, log, cfg.experiment_id)

if __name__ == "__main__":
    main()



