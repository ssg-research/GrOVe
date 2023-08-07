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
import pandas as pd
import os
import sys
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from joblib import dump, load

from model import train_model
from utils import get_model_details, read_data_npy, split_by_dataset_and_target
from feature_engineering import process_embds
pd.options.mode.chained_assignment = None  # default='warn'

def load_embeddings(embedding_dir, experiment_id):
    embeddings_dict = read_data_npy(embedding_dir, experiment_id)

    start = embedding_dir.find('False_') + len('False_')
    end = embedding_dir.rfind('/')
    attack_type = embedding_dir[start:end]

    return embeddings_dict, attack_type

def split_train_test_by_surrogate_id(embeddings, train_surrogate_id):
    train_embeddings = {}
    test_embeddings = {}

    for model_name, embeddings_dict in embeddings.items():
        _, _, _, _, _, surrogate_id, _ = get_model_details(model_name)

        if int(surrogate_id) == train_surrogate_id:
            train_embeddings[model_name] = embeddings_dict
        else:
            test_embeddings[model_name] = embeddings_dict

    return train_embeddings, test_embeddings

def get_train(embeddings, experiment_id):

    all_embeddings = [embds_tupl for embds_tupl in embeddings.values()]
    
    embds_target = np.concatenate([tupl[0] for tupl in all_embeddings], axis=0)
    embds_independent = np.concatenate([tupl[1] for tupl in all_embeddings], axis=0)
    embds_surrogate = np.concatenate([tupl[2] for tupl in all_embeddings], axis=0)

    surrogate_features, independent_features = process_embds(embds_target, embds_independent, embds_surrogate)

    train_X = np.append(surrogate_features, independent_features, axis=0)
    train_Y = np.append(np.ones(len(surrogate_features)), 
                        np.full(len(independent_features), fill_value=0), axis=0)

    train_X, train_Y = shuffle(train_X, train_Y, random_state=experiment_id)

    return train_X, train_Y

def get_ratio(clf, embeddings):
    embds_target = embeddings[0]
    embds_independent = embeddings[1]
    embds_surrogate = embeddings[2]

    surrogate_features, independent_features = process_embds(embds_target, embds_independent, embds_surrogate)

    surrogate_labels = clf.predict(surrogate_features)
    independent_labels = clf.predict(independent_features)

    surrogate_ratio = np.count_nonzero(surrogate_labels == 1) / len(surrogate_labels)
    independent_ratio = np.count_nonzero(independent_labels == 1) / len(independent_labels)

    return surrogate_ratio, independent_ratio

def get_ratios(clf, embds_dictionary, results_dictionary):
    independent_ratios = []
    surrogate_ratios = []
    for model_name, embeddings in embds_dictionary.items():
        target_model, independent_model, surrogate_model, dataset, _, surrogate_id, experiment_id = get_model_details(model_name) 
        total_embeddings = len(embeddings[0])
        surrogate_ratio, independent_ratio = get_ratio(clf, embeddings)

        independent_ratios.append(independent_ratio)
        surrogate_ratios.append(surrogate_ratio)

        results_dictionary[model_name] = {
            'experiment_id': experiment_id,
            'total_embeddings': total_embeddings,
            'target_model': target_model,
            'independent_model': independent_model,
            'surrogate_model': surrogate_model,
            'surrogate_id': surrogate_id,
            'dataset': dataset,
            'independent_ratio': independent_ratio,
            'surrogate_ratio': surrogate_ratio
        }
    
    return independent_ratios, surrogate_ratios

def make_df(results_dictionary):
    df_dict = {
        'Experiment ID': [],
        'Dataset': [],
        'Victim Model': [],
        'Independent Model': [],
        'Surrogate Model': [],
        'Surrogate ID': [],
        'Total Embeddings': [],
        'Independent Same Distribution Ratio': [],
        'Surrogate Same Distribution Ratio': []
    }

    for results in results_dictionary.values():
        df_dict['Experiment ID'].append(results['experiment_id'])
        df_dict['Dataset'].append(results['dataset'])
        df_dict['Victim Model'].append(results['target_model'])
        df_dict['Independent Model'].append(results['independent_model'])
        df_dict['Surrogate Model'].append(results['surrogate_model'])
        df_dict['Surrogate ID'].append(results['surrogate_id'])
        df_dict['Total Embeddings'].append(results['total_embeddings'])
        df_dict['Independent Same Distribution Ratio'].append(results['independent_ratio'])
        df_dict['Surrogate Same Distribution Ratio'].append(results['surrogate_ratio'])

    df = pd.DataFrame.from_dict(df_dict)

    df.sort_values(by=['Dataset', 'Victim Model', 'Independent Model', 'Surrogate Model'], inplace=True)

    return df

def test_models(attack_type, attack_embeddings, verifiers, experiment_id, results_dir):
    results_dictionary = {}
    # Split embeddings by model type
    embeddings_per_target = split_by_dataset_and_target(attack_embeddings)
    # Get predictions for each model
    for target_model, test_embeddings in embeddings_per_target.items():
        get_ratios(verifiers[target_model], test_embeddings, results_dictionary)

    # Save results for this attack in csv
    results_df = make_df(results_dictionary)
    print(results_df)
    attack_results_dir = '{}/{}/'.format(results_dir, attack_type)
    print("Saving results:", attack_results_dir)
    os.makedirs(attack_results_dir, exist_ok = True)
    results_df.to_csv(attack_results_dir+'experiment_{}.csv'.format(experiment_id), index=False)

def save_models(verifiers, results_dir, experiment_id):
    print("Saving models.")
    models_directory = results_dir + '/models/'
    os.makedirs(models_directory, exist_ok=True)
    for target_model, verifier in verifiers.items():
        filename = models_directory + target_model + '_{}.joblib'.format(experiment_id)
        dump(verifier, filename)

def load_model(model_dir, target_model, experiment_id, train_X, train_Y):
    model_filename = model_dir + 'models/{}_{}.joblib'.format(target_model, experiment_id)
    model = load(model_filename)

    preds_Y = model.predict(train_X)

    accuracy = accuracy_score(train_Y, preds_Y)
 
    return model, accuracy

def run_experiment(train_embeddings_dirs, test_embeddings_dirs, experiment_id, train_surrogate_id, results_dir):
    os.makedirs(results_dir, exist_ok = True)

    print("Loading train embeddings")
    training_data_per_model = {}
    for train_embeddings_dir in train_embeddings_dirs:
        # Load training embedding
        embeddings_dict, _ = load_embeddings(train_embeddings_dir, experiment_id)

        embeddings_per_target = split_by_dataset_and_target(embeddings_dict)

        # Each target model will have its own verification model
        for target_model, embeddings_dict in embeddings_per_target.items():
            train_embeddings_dict, _ = split_train_test_by_surrogate_id(embeddings_dict, train_surrogate_id)

            train_X, train_Y = get_train(train_embeddings_dict, experiment_id)
            if target_model not in training_data_per_model:
                training_data_per_model[target_model] = []

            training_data_per_model[target_model].append((train_X, train_Y))

    print("Train set created. Training models")

    # Train models
    accuracies = {}
    verifiers = {}
    training_times = {}
    for target_model, training_data in training_data_per_model.items():
        print("Training:", target_model)
        train_X = np.concatenate([X for X, _ in training_data])
        train_Y = np.concatenate([Y for _, Y in training_data])
        start = timer()
        verifiers[target_model], accuracies[target_model] = train_model(train_X, train_Y)
        # verifiers[target_model], accuracies[target_model] = load_model('results_robust/', target_model, experiment_id, train_X, train_Y)
        training_times[target_model] = timer() - start

    print("Models trained. Running inference.")

    # Save times for model training
    times_filename = results_dir + '/fingerprinting_times_{}.csv'.format(experiment_id)
    times_file = open(times_filename,'w')
    times_file.write('dataset,architecture,time\n')
    for target_model, time in training_times.items():
        target_model = target_model.split('_')
        dataset = target_model[0]
        architecture = target_model[1]
        times_file.write('{},{},{}\n'.format(dataset, architecture, time))

    times_file.close()

    # Test models using testing embeddings
    for test_embeddings_dir in test_embeddings_dirs:
        print("Testing on dir:", test_embeddings_dir)
        embeddings_dict, attack_type =  load_embeddings(test_embeddings_dir, experiment_id)
        _, test_embeddings = split_train_test_by_surrogate_id(embeddings_dict, train_surrogate_id)
        test_models(attack_type, test_embeddings, verifiers, experiment_id, results_dir)

    # Save the models to run inference later
    save_models(verifiers, results_dir, experiment_id)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please use the following input: python train_similarity_model.py <output_dir>")
        exit()
    else:
        results_dir = sys.argv[1]

    # Define embeddings
    train_embeddings_dirs = [
        './model_stealing/embeddings_overlap_False_embedding_original_simple_extraction/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.1/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.2/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.3/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.4/',
    ]

    test_embeddings_dirs = [
        './model_stealing/embeddings_overlap_False_embedding_original_simple_extraction/',
        './model_stealing/embeddings_overlap_False_embedding_original_double_extraction/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.1/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.2/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.3/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.4/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.5/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.6/',
        './model_stealing/embeddings_overlap_False_embedding_original_pruning_ratio_0.7/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_simple_extraction/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_double_extraction/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.1/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.2/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.3/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.4/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.5/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.6/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.7/',
        './model_stealing/embeddings_overlap_False_embedding_idgl_pruning_ratio_0.7/',
        './model_stealing/embeddings_overlap_False_embedding_original_fine_tune'
    ]

    for experiment_id in range(0, 5):
        print("*************** Experiment ID: {} ***************".format(experiment_id))
        run_experiment(train_embeddings_dirs, test_embeddings_dirs, experiment_id, 0)
    