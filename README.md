Here are other models' code:

STG-Mamba: https://github.com/LincanLi-X/STG-Mamba 
search_space = {
    "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
    "n_layer": trial.suggest_int("n_layer", 1, 6),
    "K": trial.suggest_int("K", 1, 4),
    "d_state": trial.suggest_categorical("d_state", [16, 32]),
    "expand": trial.suggest_categorical("expand", [2, 4]),
    "d_conv": trial.suggest_int("d_conv", 3, 5)
}

STGformer: https://github.com/Dreamzz5/STGformer
search_space = {
    "input_embedding_dim": trial.suggest_categorical("input_embedding_dim", [12, 24, 32, 64]),
    "tod_embedding_dim": trial.suggest_categorical("tod_embedding_dim", [8, 12, 16, 24, 32]),
    "dow_embedding_dim": trial.suggest_categorical("dow_embedding_dim", [8, 12, 16, 24, 32]),
    "adaptive_embedding_dim": trial.suggest_categorical("adaptive_embedding_dim", [12, 24, 32, 64]),
    "spatial_embedding_dim": trial.suggest_categorical("spatial_embedding_dim", [0, 12, 24, 32]),
    "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
    "num_layers": trial.suggest_int("num_layers", 1, 6),
    "mlp_ratio": trial.suggest_int("mlp_ratio", 2, 4),
    "order": trial.suggest_int("order", 1, 4),
    "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
    "dropout_a": trial.suggest_categorical("dropout_a", [0.1, 0.2, 0.3, 0.5])
}

DCRNN: https://github.com/liyaguang/DCRNN
search_space = {
    "rnn_units": trial.suggest_categorical("rnn_units", [32, 64, 128]),
    "num_rnn_layers": trial.suggest_int("num_rnn_layers", 1, 3),
    "max_diffusion_step": trial.suggest_int("max_diffusion_step", 1, 3),
    "filter_type": trial.suggest_categorical("filter_type", ["laplacian", "random_walk", "dual_random_walk"]),
    "use_curriculum_learning": trial.suggest_categorical("use_curriculum_learning", [True, False]),
    "cl_decay_steps": trial.suggest_categorical("cl_decay_steps", [1000, 2000, 3000]),
    "max_grad_norm": trial.suggest_float("max_grad_norm", 3.0, 5.0, step=1.0)
}

ASTGCN: https://github.com/wanhuaiyu/ASTGCN
search_space = {
    "K": trial.suggest_int("K", 1, 4),
    "num_of_chev_filters": trial.suggest_categorical("num_of_chev_filters", [32, 64, 128]),
    "num_of_time_filters": trial.suggest_categorical("num_of_time_filters", [32, 64, 128]),
    "time_conv_strides": trial.suggest_categorical("time_conv_strides", [1, 2]),
    "num_blocks": trial.suggest_int("num_blocks", 1, 4)
}

DDGCRN: https://github.com/wengwenchao123/DDGCRN
search_space = {
    "embed_dim": trial.suggest_categorical("embed_dim", [8, 16, 32, 64]),
    "rnn_units": trial.suggest_categorical("rnn_units", [32, 64, 128]),
    "num_layers": trial.suggest_int("num_layers", 1, 3),
    "cheb_k": trial.suggest_int("cheb_k", 1, 4),
    "use_day": trial.suggest_categorical("use_day", [True, False]),
    "use_week": trial.suggest_categorical("use_week", [True, False]),
    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    "lr_init": trial.suggest_categorical("lr_init", [1e-4, 5e-4, 1e-3, 5e-3]),
    "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 0.0]),
    "lr_decay_rate": trial.suggest_categorical("lr_decay_rate", [0.1, 0.3, 0.5, 0.7]),
    "max_grad_norm": trial.suggest_int("max_grad_norm", 3, 5)
}

GMAN: https://github.com/zhengchuanpan/GMAN
search_space = {
    "L": trial.suggest_int("L", 1, 5),
    "K": trial.suggest_categorical("K", [4, 8, 16]),
    "d": trial.suggest_categorical("d", [8, 16, 32])
}

PDFormer: original code not provided / needs to be requested
search_space = {
    "d": trial.suggest_categorical("d", [16, 32, 64, 128]),
    "L": trial.suggest_categorical("L", [2, 4, 6, 8])
}

STGHformer: original code not provided / needs to be requested
search_space = {
    "num_st_blocks": trial.suggest_int("num_st_blocks", 1, 5),
    "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8, 16]),
    "hidden_dimension": trial.suggest_categorical("hidden_dimension", [16, 24, 32, 40, 48]),
    "epochs": trial.suggest_categorical("epochs", [50, 100, 150, 200, 250, 300])
}

STMTL: original code not provided / needs to be requested
parameters fixed

EF-former: original code not provided / needs to be requested
search_space = {
    "d_model": trial.suggest_categorical("d_model", [32, 48, 64, 72]), #[cite: 63]
    "h": trial.suggest_categorical("h", [2, 4, 6, 8]) #[cite: 63]
}

MFST-GNN: original code not provided / needs to be requested
parameters fixed

STJGCN: original code not provided / needs to be requested
search_space = {
    "delta_pdf": trial.suggest_categorical("delta_pdf", [0.1, 0.3, 0.5, 0.7, 0.9]),
    "delta_adt": trial.suggest_categorical("delta_adt", [0.1, 0.3, 0.5, 0.7, 0.9]),
    "d": trial.suggest_categorical("d", [16, 32, 64, 128, 256]),
    "K": trial.suggest_categorical("K", [1, 2, 3, 4]),
    "beta": trial.suggest_categorical("beta", [0, 0.5, 1.0, 1.5, 2.0])
}

IPF-HMGNN: original code not provided / needs to be requested
search_space = {
    "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]),  #[cite: 130]
    "hidden_node": trial.suggest_categorical("hidden_node", [32, 64, 128, 256]),  #[cite: 130]
    "layer_number": trial.suggest_categorical("layer_number", [1, 2, 3]),  #[cite: 130]
    "dropout": trial.suggest_float("dropout", 0.0, 0.5),  #[cite: 130]
    "A": trial.suggest_categorical("A", [1, 2, 3, 4, 5, 6, 7, 8]),  #[cite: 122]
    "K": trial.suggest_categorical("K", [2, 3, 4, 5, 6])  #[cite: 122]
}

GRU-Informer: original code not provided / needs to be requested
search_space = {
    "batch_size": trial.suggest_categorical("batch_size", [16, 28, 40, 52, 64]),
    "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]),
    "drop_rate": trial.suggest_categorical("drop_rate", [0.1, 0.15, 0.2, 0.25, 0.3])
}

We don't have permission to share the EXO dataset;
Similar datasets are available on request;

To make the paper more reproducible, we’ve organized and released the code.
The code is organized into separate modules: data_processing.py handles data preprocessing; graph_construction.py constructs the graphs; parameters.py manages configuration and survey-based parameter calibration; model.py implements the model architecture; training.py handles training and hyperparameter grid search; main.py serves as the main entry point; indicators.py computes evaluation metrics; experiments.py manages repeated experiments, paired statistical tests, and inference timing; data_audit.py checks dataset statistics across all regions; and smoke_test.py and test_case_protocol.py provide implementation and experimental protocol checks.


The final results for all regions can be found.

If you have any problem, please send email to xinyizhou0318@gmail.com
