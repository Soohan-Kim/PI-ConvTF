from datetime import datetime as dt

## Train & Test Periods
train_start = dt(2004, 1, 5)
train_end = dt(2019, 12, 31)
test_start = dt(2020, 1, 1)
test_end = dt(2021, 8, 13)

## Common (low, high) range to rescale underlying price and strike into
scale_low = 0.01
scale_high = 1

'''
Following are hyperparameters for model training and architecture.

For PINN optimizer choice, string denoting optimizer name should be specified. (as according to SciANN package)
'''

## PINN Model Hyperparameters
PINN = {
    'epochs': 2000,
    'batch_size': 256,
    'optimizer': 'Adam',
    'init_lr': 0.1,
    'reduce_lr_after': 20,
    'stop_lr_value': 1e-10,
    'PINN_range': [-80000, -40000],
    'model_hp': {
        'volatility': {
            'hidden_layers': [10000],
            'activation': 'softplus'
        },
        'call': {
            'hidden_layers': [10000],
            'activation': 'softplus'
        }
    },
    'model_load_path': None, # If conducting transfer learning
    'model_save_name': ''
}
