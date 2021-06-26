"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'paths': {
            'train': str,
            'test': str,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'hidden_bb_dim': int,
        'hidden_label_dim': int,
        'bb_loss_weight': int,
        'batch_size': int,
        'save_model': bool,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
