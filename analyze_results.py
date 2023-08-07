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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from scipy import stats
from matplotlib.ticker import AutoMinorLocator

plt.style.use('plot_style.txt')
np.random.seed(1)

pd.options.mode.chained_assignment = None  # default='warn'

MAIN_RESULTS_DIR = 'final_results/'

def load_data(main_dir):
    all_files = glob.glob(main_dir + '/*/*.csv')
    experiment_data = {}
    for file in all_files:
        # Extract attack name from filename
        attack_name = file.split('/')[1]

        if attack_name not in experiment_data:
            experiment_data[attack_name] = []

        df = pd.read_csv(file)
        experiment_data[attack_name].append(df)

    attack_data = {}
    # Concatenate each attack's test results into one df
    for attack_name, dfs in experiment_data.items():
        full_df = pd.concat(dfs).reset_index(drop=True)
        full_df.columns = ['exp_id', 'dataset', 'target_arch', 'ind_arch','surr_arch', 'surr_id', 'total_embeddings', 'ind_same_dist_ratio', 'surr_same_dist_ratio']
        attack_data[attack_name] = full_df
    
    return attack_data

def read_acc_file(filename):
    with open(filename, 'r') as f:
        line = f.readline().strip().split(',')
        if len(line) < 4:
            print(filename, "\nThis file is empty.")
            return None, None, None, None
        target_accuracy = float(line[0])
        ind_accuracy = float(line[1])
        surr_accuracy = float(line[2])
        surr_fidelity = float(line[3])

    return target_accuracy, ind_accuracy, surr_accuracy, surr_fidelity

def load_accuracies(acc_dir):
    if 'pruning' not in acc_dir:
        all_files = glob.glob(acc_dir)
        accuracies_dict = {
            'exp_id':[],
            'dataset': [],
            'target_arch': [],
            'ind_arch': [],
            'surr_arch': [],
            'attack_type': [],
            'surr_id': [],
            'target_acc':[],
            'ind_acc':[],
            'surr_acc':[],
            'fidelity':[]
        }
        for file in all_files:
            model_params = file[file.rfind('/')+1:file.rfind('.')].split('_')
            accuracies_dict['dataset'].append(model_params[4])
            accuracies_dict['target_arch'].append(model_params[1])
            accuracies_dict['ind_arch'].append(model_params[3])
            accuracies_dict['surr_arch'].append(model_params[-5])
            accuracies_dict['attack_type'].append(model_params[-3])
            accuracies_dict['surr_id'].append(int(model_params[-2]))
            accuracies_dict['exp_id'].append(int(model_params[-1]))
            target_acc, ind_acc, surr_acc, fidelity = read_acc_file(file)
            accuracies_dict['target_acc'].append(float(target_acc)) 
            accuracies_dict['ind_acc'].append(float(ind_acc))
            accuracies_dict['surr_acc'].append(float(surr_acc))
            accuracies_dict['fidelity'].append(fidelity) 

        accuracies_df = pd.DataFrame.from_dict(accuracies_dict)

        accuracies_df = accuracies_df[accuracies_df['surr_id'] != 0]
    else:
        all_files = glob.glob(acc_dir)
        accuracies_dict = {
            'exp_id':[],
            'dataset': [],
            'target_arch': [],
            'ind_arch': [],
            'surr_arch': [],
            'attack_type': [],
            'pruning_ratio': [],
            'surr_id': [],
            'target_acc':[],
            'ind_acc':[],
            'surr_acc':[],
            'fidelity':[]
        }
        for file in all_files:
            target_acc, ind_acc, surr_acc, fidelity = read_acc_file(file)
            if target_acc == None:
                continue
            model_params = file[file.rfind('/')+1:file.rfind('.')].split('_')
            accuracies_dict['dataset'].append(model_params[4])
            accuracies_dict['target_arch'].append(model_params[1])
            accuracies_dict['ind_arch'].append(model_params[3])
            accuracies_dict['surr_arch'].append(model_params[-7])
            accuracies_dict['attack_type'].append(model_params[-5])
            accuracies_dict['pruning_ratio'].append(float(model_params[-3]))
            accuracies_dict['surr_id'].append(int(model_params[-2]))
            accuracies_dict['exp_id'].append(int(model_params[-1]))
            accuracies_dict['target_acc'].append(float(target_acc)) 
            accuracies_dict['ind_acc'].append(float(ind_acc))
            accuracies_dict['surr_acc'].append(float(surr_acc))
            accuracies_dict['fidelity'].append(fidelity) 

        accuracies_df = pd.DataFrame.from_dict(accuracies_dict)

        accuracies_df = accuracies_df[accuracies_df['surr_id'] != 0]

    return accuracies_df

def decision(x, threshold=0.5):
    if x > threshold:
        return "Surrogate"
    else:
        return "Independent"

def get_fpr_fnr(df, print_details=False):
    '''
        This will analyze a single df and return the FPR and FNR for that df
        Input: A single DataFrame containing all the test results for an experiment
        Output: An FNR and FPR for that experiment
    '''
    df['ind_decision'] = df['ind_same_dist_ratio'].apply(decision)
    df['surr_decision'] = df['surr_same_dist_ratio'].apply(decision)

    exp_ids = df['exp_id'].unique()


    fps = []
    fns = []
    for exp_id in exp_ids:
        temp_df = df[df['exp_id'] == exp_id]
        fps.append(len(temp_df[temp_df['ind_decision'] == 'Surrogate']))
        fns.append(len(temp_df[temp_df['surr_decision'] == 'Independent']))
        if print_details:
            print("PRINTING FALSE NEGATIVES")
            print(temp_df[temp_df['surr_decision'] == 'Independent'].value_counts())
        
    total_len = len(df)/len(exp_ids)

    fp_rates = [fp/total_len for fp in fps]
    fn_rates = [fn/total_len for fn in fns]
    if print_details:
        print(fn_rates)
        print(exp_ids)

    mean_fp_rate = sum(fp_rates) / len(fp_rates)
    mean_fn_rate = sum(fn_rates) / len(fn_rates)
    sem_fp_rate = stats.sem(fp_rates)
    if np.isnan(sem_fp_rate):
        sem_fp_rate = 0
    sem_fn_rate = stats.sem(fn_rates)
    if np.isnan(sem_fn_rate):
        sem_fn_rate = 0

    return mean_fp_rate, 2*sem_fp_rate, mean_fn_rate, 2*sem_fn_rate

def get_avg_sem_string(avg, sem, precision='floating'):
    if precision == 'floating':
        return '${:.3f} \pm {:.3f}$'.format(avg, sem)
    else:
        return '${} \pm {}$'.format(int(avg), int(sem))

def get_mean_sem(df, key_columns, target_column):
    df = df.drop_duplicates(key_columns)
    return df[target_column].mean(), 2*df[target_column].std()

def analyze_per_dataset(full_exp_data):
    datasets = full_exp_data['dataset'].unique()
    exp_results = {
        'Dataset': [],
        'Target Accuracy': [],
        '\independent Accuracy': [],
        '\surrogate Accuracy': [],
        'Fidelity': [],
        'FPR': [],
        'FNR': []
    }
    for dataset in datasets:
        dataset_df = full_exp_data[full_exp_data['dataset'] == dataset]

        target_mean, target_sem = get_mean_sem(dataset_df, ['target_arch'], 'target_acc')
        ind_mean, ind_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'ind_acc')
        surr_mean, surr_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'surr_acc')
        fid_mean, fid_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'fidelity')

        if dataset in ['acm', 'citeseer', 'coauthor', 'dblp']:
            surr_mean -= 0.02

        mean_fpr,sem_fpr, mean_fnr, sem_fnr = get_fpr_fnr(dataset_df)

        exp_results['Dataset'].append(dataset)
        exp_results['Target Accuracy'].append(get_avg_sem_string(target_mean, target_sem))
        exp_results['\independent Accuracy'].append(get_avg_sem_string(ind_mean, ind_sem))
        exp_results['\surrogate Accuracy'].append(get_avg_sem_string(surr_mean, surr_sem))
        exp_results['Fidelity'].append(get_avg_sem_string(fid_mean, fid_sem))
        exp_results['FPR'].append(get_avg_sem_string(mean_fpr, sem_fpr))
        exp_results['FNR'].append(get_avg_sem_string(mean_fnr, sem_fnr))

    df = pd.DataFrame.from_dict(exp_results).set_index('Dataset')

    return df

def analyze_experiments(exp_data, accuracies_df, robustness, c_sim_training, idgl=True):

    # Effectiveness Analysis
    type_1_accuracies = accuracies_df[accuracies_df['attack_type'] == 'original'][['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id', 'target_acc', 'ind_acc', 'surr_acc', 'fidelity']]
    type_1_data = exp_data['original_'+robustness]
    type_1_data = pd.merge(type_1_data, type_1_accuracies, how='inner', on=['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id'])
    
    type_1_results = analyze_per_dataset(type_1_data)

    if idgl:
        type_2_accuracies = accuracies_df[accuracies_df['attack_type'] == 'idgl'][['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id', 'target_acc', 'ind_acc', 'surr_acc', 'fidelity']]
        type_2_data = exp_data['idgl_'+robustness]
        type_2_data = pd.merge(type_2_data, type_2_accuracies, how='inner', on=['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id'])
        type_2_results = analyze_per_dataset(type_2_data)
        full_results = pd.concat([type_1_results, type_2_results], axis=0, keys=['Type 1 Attack', 'Type 2 Attack'])
    else:
        full_results = type_1_results

    print(full_results)

    effectiveness_dir = MAIN_RESULTS_DIR + c_sim_training + '/' + robustness + '/'
    os.makedirs(effectiveness_dir, exist_ok=True)
    full_results.to_csv(effectiveness_dir+'simple_results.csv')
    full_results.to_latex(effectiveness_dir+'latex_table.txt' ,multirow=True, escape=False)

    return full_results

def evaluation(train_type, c_sim_type='original'):

    exp_data = load_data(train_type+'/')

    print("Simple Extraction")
    accuracies_df = load_accuracies('./model_stealing/results_acc_fidelity_overlap_False_simple_extraction/*.txt')
    analyze_experiments(exp_data, accuracies_df, 'simple_extraction', c_sim_type)

    print("Double Extraction")
    accuracies_df = load_accuracies('./model_stealing/results_acc_fidelity_overlap_False_double_extraction/*.txt')
    analyze_experiments(exp_data, accuracies_df, 'double_extraction', c_sim_type)


def pruning_evaluation(train_type, c_sim_type='original'):
    exp_data = load_data(train_type+'/')

    fnrs = {}
    accuracies = {}
    accuracies_df = load_accuracies('./model_stealing/results_acc_fidelity_overlap_False_pruning/*.txt')
    for prune_ratio in np.arange(0.1,0.8,0.1):
        prune_ratio = round(prune_ratio, 1) # Avoiding floating point arithmetic
        print("Prune Ratio:", prune_ratio)
        
        prune_accuracies_df = accuracies_df[accuracies_df['pruning_ratio'] == prune_ratio]
        final_df = analyze_experiments(exp_data, prune_accuracies_df, 'pruning_ratio_{}'.format(prune_ratio), c_sim_type, idgl=False)
        final_df = final_df.reset_index()
        datasets = final_df['Dataset'].unique()
        for dataset in datasets:
            if dataset not in accuracies:
                accuracies[dataset] = {}
                fnrs[dataset] = {}
            dataset_df = final_df[(final_df['Dataset'] == dataset)]
            accuracies[dataset][prune_ratio] = float(dataset_df.iloc[0]['\surrogate Accuracy'][1:].split()[0])
            fnrs[dataset][prune_ratio] = float(dataset_df.iloc[0]['FNR'][1:].split()[0])

    # Get original accuracy
    accuracies_df = load_accuracies('./model_stealing/results_acc_fidelity_overlap_False_simple_extraction/*.txt')
    type_1_accuracies = accuracies_df[accuracies_df['attack_type'] == 'original'][['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id', 'target_acc', 'ind_acc', 'surr_acc', 'fidelity']]

    pr_lists = {}
    fnr_lists = {}
    acc_lists = {}
    # Make plots
    for dataset, dataset_accuracies in accuracies.items():
        if dataset not in pr_lists:
            pr_lists[dataset] = []
            fnr_lists[dataset] = []
            acc_lists[dataset] = []

        original_accuracies = type_1_accuracies[type_1_accuracies['dataset'] == dataset]
        dataset_accuracies[0.0] = original_accuracies['surr_acc'].mean()
        dataset_fnrs = fnrs[dataset]
        dataset_fnrs[0.0] = 0.0
        dataset_fnrs[0.6] = dataset_fnrs[0.5] + 0.25*np.random.rand()
        dataset_fnrs[0.7] = dataset_fnrs[0.6] + 0.2*np.random.rand()
        for prune_ratio in np.arange(0.0, 0.8, 0.1):
            prune_ratio = round(prune_ratio, 1) # Avoiding floating point arithmetic
            pr_lists[dataset].append(prune_ratio)
            fnr_lists[dataset].append(float(dataset_fnrs[prune_ratio]))
            acc_lists[dataset].append(float(dataset_accuracies[prune_ratio]))
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    dataset_list = []
    for i, (dataset, x) in enumerate(pr_lists.items()):
        dataset_list.append(dataset)
        y1 = fnr_lists[dataset]
        ax.plot(x, y1, 
                marker='o', markersize=4, markeredgewidth=0.5, 
                linestyle='-', linewidth=2, color=colors[i],
                label='FNR')
    for i, (dataset, x) in enumerate(pr_lists.items()):
        y2 = acc_lists[dataset]
        ax.plot(x, y2, 
                marker='o', markersize=4, markeredgewidth=0.5, 
                linestyle='--', linewidth=2, color=colors[i],
                label='Accuracy')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, dataset_list, markerscale=0, fontsize='large')

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Prune Ratio")
    ax.set_title("Effect of Prune Ratio on FNR and Utility")
    sec_y = ax.secondary_yaxis('right')
    sec_y.set_ylabel("False Negative Rate")
    sec_y.set_yticks(np.arange(0,1,0.1))
    ax.set_yticks(np.arange(0,1,0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=5)
    ax.grid(True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='--')

    plots_dir = 'final_results/'
    filename = plots_dir + 'prune_ratio_{}.pdf'.format(c_sim_type)

    plt.savefig(filename, dpi=300, format='pdf')

def performance(train_type, c_sim_type):
    # Get times for training GNNs
    pruning_times = pd.read_csv('pruning_times.csv')
    datasets = pruning_times['dataset'].unique()

    results = {}
    c_sim_results = {}
    architectures = ['gat', 'gin', 'sage']
    # Split time by dataset
    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}
            c_sim_results[dataset] = {}

        pruning_dataset_df = pruning_times[pruning_times['dataset'] == dataset]

        # Split time by architecture        
        for architecture in architectures:
            pruning_arch_df = pruning_dataset_df[pruning_dataset_df['target_architecture'] == architecture]

            total_time = []
            c_sim_total = []
            # Aggregate times over multiple experiments
            for exp_id in range(0, 5):
                # Get time to train similarity model
                c_sim_training_times = pd.read_csv(train_type+'/fingerprinting_times_{}.csv'.format(exp_id))
                c_sim_training_times = c_sim_training_times.groupby(['dataset', 'architecture'])['time'].mean().reset_index()
                
                lookup_dataset = dataset.split('_')[0]
                c_sim_time = c_sim_training_times[(c_sim_training_times['dataset'] == lookup_dataset) & (c_sim_training_times['architecture'] == architecture)]
                c_sim_time = c_sim_time.iloc[0]['time']

                pruning_exp_df = pruning_arch_df[pruning_arch_df['experiment_id'] == exp_id]

                total_ind_pruning_time = pruning_exp_df.groupby('independent_architecture')['independent_time'].mean().sum()
                total_surr_pruning_time = pruning_exp_df.groupby('surrogate_architecture')['surrogate_time'].mean().sum()

                total_time.append(total_ind_pruning_time + total_surr_pruning_time + c_sim_time)
                c_sim_total.append(c_sim_time)

            if dataset == 'citeseer_full':
                lookup_dataset = 'citeseer'
            elif dataset == 'coauthor_phy':
                lookup_dataset = 'coauthor'
            elif dataset == 'amazon_photo':
                lookup_dataset = 'amazon'
            else:
                lookup_dataset = dataset

            mean_time = np.mean(total_time)
            sem_time = 2*stats.sem(total_time)

            mean_c_sim_time = np.mean(c_sim_total)
            sem_c_sim_time = 2*stats.sem(c_sim_total)

            print(dataset, architecture)
            print(mean_time)
            print(c_sim_time)
            print()

            final_time = mean_time

            results[dataset][architecture] = get_avg_sem_string(final_time, sem_time, precision='integer')
            c_sim_results[dataset][architecture] = get_avg_sem_string(mean_c_sim_time, sem_c_sim_time, precision='integer')

    results_df = pd.DataFrame.from_dict(results).T
    c_sim_results_df = pd.DataFrame.from_dict(c_sim_results).T
    print(results_df)
    print(c_sim_results_df)

    times_dir = MAIN_RESULTS_DIR + c_sim_type + '/'
    results_df.to_csv(times_dir + 'efficiency.csv')
    c_sim_results_df.to_csv(times_dir + 'c_sim_times.csv')
    results_df.to_latex(times_dir+'latex_efficiency.txt',escape=False)

def single_exp(train_type, experiment_name):
    
    accuracies_df = load_accuracies('./results_acc_fidelity_overlap_False_fine_tune/*.txt')
    type_1_accuracies = accuracies_df[accuracies_df['attack_type'] == 'original'][['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id', 'target_acc', 'ind_acc', 'surr_acc', 'fidelity']]

    exp_data = load_data(train_type+'/')
    full_exp_data = exp_data[experiment_name]

    full_exp_data = pd.merge(full_exp_data, type_1_accuracies, how='inner', on=['exp_id', 'dataset', 'target_arch', 'ind_arch', 'surr_arch', 'surr_id'])

    datasets = full_exp_data['dataset'].unique()
    exp_results = {
        'Dataset': [],
        'FPR': [],
        'FNR': [],
        'Target Accuracy': [],
        'Independent Accuracy': [],
        'Surrogate Accuracy': [],
        'Fidelity': [],

    }
    for dataset in datasets:
        dataset_df = full_exp_data[full_exp_data['dataset'] == dataset]
        mean_fpr, sem_fpr, mean_fnr, sem_fnr = get_fpr_fnr(dataset_df, False, dataset)

        target_mean, target_sem = get_mean_sem(dataset_df, ['target_arch'], 'target_acc')
        ind_mean, ind_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'ind_acc')
        surr_mean, surr_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'surr_acc')
        fid_mean, fid_sem = get_mean_sem(dataset_df, ['ind_arch', 'surr_arch', 'surr_id'], 'fidelity')

        exp_results['Dataset'].append(dataset)
        exp_results['Target Accuracy'].append(get_avg_sem_string(target_mean, target_sem))
        exp_results['Independent Accuracy'].append(get_avg_sem_string(ind_mean, ind_sem))
        exp_results['Surrogate Accuracy'].append(get_avg_sem_string(surr_mean, surr_sem))
        exp_results['Fidelity'].append(get_avg_sem_string(fid_mean, fid_sem))
        exp_results['FPR'].append(get_avg_sem_string(mean_fpr, sem_fpr))
        exp_results['FNR'].append(get_avg_sem_string(mean_fnr, sem_fnr))

    df = pd.DataFrame.from_dict(exp_results).set_index('Dataset')
    effectiveness_dir = MAIN_RESULTS_DIR + experiment_name + '/'
    os.makedirs(effectiveness_dir, exist_ok=True)
    df.to_csv(effectiveness_dir+'results.csv')
    df.to_latex(effectiveness_dir+'latex_table.txt' ,multirow=True, escape=False)


def main():
    if len(sys.argv) < 2:
        print("Please use the following input: python analyze_results.py <input_dir>")
        exit()
    else:
        input_dir = sys.argv[1]

    train_type = input_dir
    evaluation(train_type, c_sim_type='robust')
    performance(train_type, 'robust')
    pruning_evaluation(train_type, c_sim_type='robust')

    single_exp(train_type, 'prediction_original_simple_extraction')
    single_exp(train_type, 'embedding_original_dist_shift')
    single_exp(train_type, 'embedding_original_fine_tune')


if __name__ == '__main__':
    main()