config = {'data_type': 'dgl',
          'dgl_graph': "placeholder",
          'dataset_name': 'cora',
          'data_dir': '../data/cora/',
          'pretrained': None,
          'task_type': 'classification',
          'out_dir': '../out/cora/idgl_del_edges',
          'out_raw_learned_adj_path': 'learned_adj.npy',
          'seed': 42,
          'n_train': 50,
          'n_val': 100,
          'model_name': 'GraphClf',
          'hidden_size': 256,
          'input_graph_knn_size': 24,
          'use_bert': False,
          'dropout': 0.5,
          'feat_adj_dropout': 0.5,
          'gl_dropout': 0,
          'bignn': False,
          'graph_module': 'graphsage',
          'graph_type': 'dynamic',
          'graph_learn': True,
          'graph_metric_type': 'weighted_cosine',
          'graph_include_self': False,
          'graph_skip_conn': 0.4,
          'update_adj_ratio': 0.1,
          'graph_learn_regularization': True,
          'smoothness_ratio': 0.4,
          'degree_ratio': 0.1,
          'sparsity_ratio': 0,
          'graph_learn_ratio': 0,
          'graph_learn_hidden_size': 20,
          'graph_learn_epsilon': 0.65,
          'graph_learn_topk': None,
          'graph_learn_hidden_size2': 20,
          'graph_learn_epsilon2': 0.65,
          'graph_learn_topk2': None,
          'graph_learn_num_pers': 8,
          'graph_hops': 2,
          'gat_nhead': 8,
          'gat_alpha': 0.2,
          'graphsage_agg_type': 'gcn',
          'optimizer': 'adam',
          'learning_rate': 0.001,
          'weight_decay': 0.0005,
          'lr_patience': 2,
          'lr_reduce_factor': 0.5,
          'grad_clipping': None,
          'grad_accumulated_steps': 1,
          'eary_stop_metric': 'nloss',
          'pretrain_epoch': 0,
          'max_iter': 0,
          'eps_adj': 0,
          'rl_ratio': 0,
          'rl_ratio_power': 1,
          'rl_start_epoch': 1,
          'max_rl_ratio': 0.99,
          'rl_reward_metric': 'acc',
          'rl_wmd_ratio': 0,
          'random_seed': 1234,
          'shuffle': True,
          'max_epochs': 10000,
          'patience': 400,
          'verbose': 20,
          'print_every_epochs': 50,
          'out_predictions': False,
          'save_params': True,
          'logging': True,
          'no_cuda': False,
          'cuda_id': 0}