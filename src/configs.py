from datetime import datetime as dt
from torch import optim

## Chosen Data Formats
m = [0.9, 1.1]
ttm = [0.04, 1.0]
g = 20

## Train & Test Periods
train_start = dt(2004, 1, 5)
train_end = dt(2019, 12, 31)
test_start = dt(2020, 1, 1)
test_end = dt(2021, 8, 13)

## Common (low, high) range to rescale underlying price and strike into
scale_low = 0.01
scale_high = 1

timestep = 10

## Common valid split ratio (train: valid)
valid_split = 0.2

## Common (low, high) range to rescale underlying price and strike into
scale_low = 0.01
scale_high = 1

'''
Following are hyperparameters for model training and architecture.

For PINN optimizer choice, string denoting optimizer name should be specified. (as according to SciANN package)

Hyperparameters regarding model architectures of convolution-based models
need to be configured such that the output is of shape (batch_size, 1, num_ttm, num_moneyness)

It is recommended to keep the input size (height and width dimensions) same 
throughout convolution operations. (i.e. configure kernel size, padding, and stride
such that height and width dimensions are maintained and only the channel dimension,
or number of filters changes)

Note that optimizer settings should be specified as a torch.optim algorithm class for ConvLSTM, SA-ConvLSTM, ConvTF, PI-ConvTF.
'''

## PINN Model Hyperparameters
PINN = {
    'epochs': 1000,
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
    'model_load_path': '../models/models/PINN_Model1-Covid.hdf5', # If conducting transfer learning
    'model_save_name': ''
}

## ConvLSTM Model Hyperparameters
ConvLSTM = {
    'epochs': 100,
    'batch_size': 32,
    'init_lr': 0.001,
    'reduce_lr_plateau': {
        'mode': 'min', 'factor': 0.5, 'threshold_mode': 'abs', 'threshold': 0.01, 'patience': 5
    },
    'model_hp': {
        'num_layers': 1,
        'conv_lstm_hp': {
            'filters': [64],
            'kernel_size': [3],
            'strides': [1],
            'padding': [0]
        },
        'last_conv_hp': {
            'kernel_size': 1,
            'stride': 1,
            'padding': 1
        }
    },
    'model_load_path': None,
    'model_save_name': ''
}
convlstm = ConvLSTM['model_hp']
assert len(convlstm['conv_lstm_hp']['filters']) == convlstm['num_layers']
assert len(convlstm['conv_lstm_hp']['kernel_size']) == convlstm['num_layers']
assert len(convlstm['conv_lstm_hp']['strides']) == convlstm['num_layers']
assert len(convlstm['conv_lstm_hp']['padding']) == convlstm['num_layers']

## SA-ConvLSTM Model Hyperparameters
SAConvLSTM = {
    'epochs': 100,
    'batch_size': 32,
    'init_lr': 0.001,
    'reduce_lr_plateau': {
        'mode': 'min', 'factor': 0.5, 'threshold_mode': 'abs', 'threshold': 0.01, 'patience': 5
    },
    'model_hp': {
        'num_layers': 1,
        'sa_conv_hp': {
            'filters': [64],
            'kernel_size': [3],
            'strides': [1],
            'padding': [0],
            'QK_channels': [8]
        },
        'last_conv_hp': {
            'kernel_size': 1,
            'stride': 1,
            'padding': 1
        }
    },
    'model_load_path': None,
    'model_save_name': ''
}

saconvlstm = SAConvLSTM['model_hp']
assert len(saconvlstm['sa_conv_hp']['filters']) == saconvlstm['num_layers']
assert len(saconvlstm['sa_conv_hp']['kernel_size']) == saconvlstm['num_layers']
assert len(saconvlstm['sa_conv_hp']['strides']) == saconvlstm['num_layers']
assert len(saconvlstm['sa_conv_hp']['padding']) == saconvlstm['num_layers']
assert len(saconvlstm['sa_conv_hp']['QK_channels']) == saconvlstm['num_layers']

## ConvTF Model Hyperparameters
ConvTF = {
    'epochs': 100,
    'batch_size': 16,
    'init_lr': 0.001,
    'reduce_lr_plateau': {
        'mode': 'min', 'factor': 0.5, 'threshold_mode': 'abs', 'threshold': 0.01, 'patience': 5
    },
    'model_hp': {
        'num_layers': 1,
        'f_chan': [4, 8, 16, 32],
        'attention_heads': 4,
        'd_model': 32,
        'sffn_config': [
            [32, 64], [64, 64], [64, 128], [128, 128], [128, 64], [64, 64], [64, 32], [32, 32],
            [32, 16], [16, 8], [8, 1],
            [1, 8], [8, 8], [8, 16], [16, 16], [16, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128],
            [128, 64], [64, 64], [64, 32], [32, 32], [32, 16], [16, 16], [16, 8], [8, 8], [8, 1]
        ]
    },
    'model_load_path': None,
    'model_save_name': ''
}

## PI-ConvTF Model Hyperparameters
PIConvTF = {
    'epochs': 100,
    'batch_size': 16,
    'init_lr': 0.001,
    'reduce_lr_plateau': {
        'mode': 'min', 'factor': 0.5, 'threshold_mode': 'abs', 'threshold': 0.01, 'patience': 5
    },
    'model_hp': {
        'num_layers': 1,
        'f_chan': [4, 8, 16, 32],
        'attention_heads': 4,
        'd_model': 32,
        'sffn_config': [
            [32, 64], [64, 64], [64, 128], [128, 128], [128, 64], [64, 64], [64, 32], [32, 32],
            [32, 16], [16, 8], [8, 1],
            [1, 8], [8, 8], [8, 16], [16, 16], [16, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128],
            [128, 64], [64, 64], [64, 32], [32, 32], [32, 16], [16, 16], [16, 8], [8, 8], [8, 1]
        ],
        'pinn_loss_weight': 0.1
    },
    'model_load_path': None,
    'model_save_name': ''
}

## Optimizer choice - change options directly inside function to use other optimizer settings
def set_optim(model_params, model_name):
    if model_name == 'ConvLSTM':
        init_lr = ConvLSTM['init_lr']
    if model_name == 'SA-ConvLSTM':
        init_lr = SAConvLSTM['init_lr']
    if model_name == 'ConvTF':
        init_lr = ConvTF['init_lr']
    if model_name == 'PI-ConvTF':
        init_lr = PIConvTF['init_lr']

    opt = optim.Adam(
        model_params,
        lr=init_lr
    )

    return opt