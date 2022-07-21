import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter

import configs
import networks

# Helper function for PI-ConvTF Gradient Calculation for Pinn Loss
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads

# Helper function for evaluating call option price from predicted volatility
def call_price(s, k, ttm, r, sigma, device):
    d1 = torch.div(torch.log(torch.div(s, k)) + (r + 0.5 * torch.square(sigma)) * ttm, sigma * torch.sqrt(ttm))
    d2 = torch.div(torch.log(torch.div(s, k)) + (r - 0.5 * torch.square(sigma)) * ttm, sigma * torch.sqrt(ttm))

    dist = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    cdf1, cdf2 = dist.cdf(d1), dist.cdf(d2)

    call = s * cdf1 - k * torch.exp(-r * ttm) * cdf2

    return call

# Helper function for PI-ConvTF Pinn Loss
def BS_PDE(sig_pred, add_data, device):
    s, k, ttm, r = add_data[:, 1, :, :], add_data[:, 3, :, :], add_data[:, 0, :, :], add_data[:, 2, :, :]
    s, k, ttm, r = torch.unsqueeze(s, dim=1), torch.unsqueeze(k, dim=1), torch.unsqueeze(ttm, dim=1), torch.unsqueeze(r, dim=1)
    c = call_price(s, k, ttm, r, sig_pred, device)

    c_t = nth_derivative(c, ttm, 1)
    c_s = nth_derivative(c, s, 1)
    c_ss = nth_derivative(c, s, 2)

    f = c_t + r*c - r*s*c_s - torch.square(sig_pred*s)*c_ss*0.5
    return f

# Helper function for MAPE calculation
## returns mape values as flattened torch tensor
def MAPE(x, y):
    x, y = x.flatten(), y.flatten()
    mapes = torch.div(100*abs(x - y), y)
    return mapes

# Trainer for ConvLSTM, SA-ConvLSTM, ConvTF, PI-ConvTF
def train(model_name, model_configs, m_range, ttm_range, grid, conv_more_inputs, data, device, pi_add_data=None):
    '''
    Trains model according to model_name and model_configs.
    Logs training information to tensorboard and saves model while training.
    If conducting transfer learning, loads model from 'model_load_path',
    resumes training, and saves it to path created from 'model_save_name'.
    If conv_more_inputs flag is set, model with additional data will be trained
    and saved to path with 'model_save_name' with prefix 'more_inputs_'.
    Returns trained model upon completion of training.

    inputs
        -model_name: name, or architecture type of model to train; one of 'ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF' (str)
        -model_configs: hyperparameter configurations of model to train (dict)
        -m_range: [moneyness low, moneyness high] (list)
        -ttm_range: [ttm low, ttm high] (list)
        -grid: grid size (int)
        -conv_more_inputs: flag for training with more inputs (bool)
        -data: [train_set, valid_set] (list of torch DataLoaders)
        -device: device to train model on (str)
        -pi_add_data: additional data for calculating pinn loss for PI-ConvTF (torch DataLoader)

    returns
        -model: trained torch model
        -model_save_name: chosen file save name for model (str)
        -DEVICE: chosen device that model trained on; will be used for test (str)
    '''
    input_channels = 1
    more_inputs = False
    if conv_more_inputs:
        model_save_name == 'more_inputs_' + model_configs['model_save_name']
        input_channels = 5
        more_inputs = True
    
    log_dir = 'logs/' + model_save_name
    writer = SummaryWriter(log_dir)

    if device[:-2] == 'gpu' and torch.cuda.is_available():
        torch.cuda.set_device(int(device[-1]))
        DEVICE = torch.device('cuda:' + device[-1])
        print('current cuda device:', torch.cuda.current_device())
    else:
        DEVICE = "cpu"
    print('\n')
    print('-GPU CHECK-')
    print(DEVICE)
    print('\n')

    model_hp = model_configs['model_hp']

    if model_name == 'ConvLSTM':
        model = networks.SAConvLSTM(
            input_channels=input_channels,
            feature_channels=model_hp['conv_lstm_hp']['filters'],
            inter_channels=[None for _ in range(model_hp['num_layers'])],
            kernel_size=model_hp['conv_lstm_hp']['kernel_size'],
            stride=model_hp['conv_lstm_hp']['strides'],
            padding=model_hp['conv_lstm_hp']['padding'],
            device=DEVICE,
            last_conv=list(model_hp['last_conv_hp'].values()),
            num_layers=model_hp['num_layers']
        ).to(DEVICE)

    elif model_name == 'SA-ConvLSTM':
        model = networks.SAConvLSTM(
            input_channels=input_channels,
            feature_channels=model_hp['sa_conv_hp']['filters'],
            inter_channels=model_hp['sa_conv_hp']['QK_channels'],
            kernel_size=model_hp['sa_conv_hp']['kernel_size'],
            stride=model_hp['sa_conv_hp']['strides'],
            padding=model_hp['sa_conv_hp']['padding'],
            device=DEVICE,
            last_conv=list(model_hp['last_conv_hp'].values()),
            num_layers=model_hp['num_layers']
        ).to(DEVICE)

    elif model_name == 'ConvTF' or model_name == 'PI-ConvTF':
        model = networks.ConvTransformer(
            more_inputs=more_inputs,
            seq_len=configs.timestep,
            channels_list=model_hp['f_chan'],
            num_layers=model_hp['num_layers'],
            num_heads=model_hp['attention_heads'],
            d_model=model_hp['d_model'],
            hidden_size=grid,
            sffn_config=model_hp['sffn_config']
        ).to(DEVICE)

        if model_name == 'PI-ConvTF':
            assert pi_add_data is not None

    else:
        print('NOT A VALID MODEL')
        quit()

    lr_hp = model_configs['reduce_lr_on_plateau']
    optimizer = configs.set_optim(model.parameters(), model_name)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=lr_hp['mode'], factor=lr_hp['factor'], threshold_mode=lr_hp['threshold_mode'],
        threshold=lr_hp['threshold'], verbose=True, patience=lr_hp['patience']
    )
    loss_func = nn.L1Loss()

    if model_configs['model_load_path'] is not None:
        model.load_state_dict(torch.load(model_configs['model_load_path']))

    train_set, valid_set = data[0], data[1]

    train_losses, valid_losses = [], []
    best_val_loss = 100
    for epoch in range(model_configs['epochs']):
        model.train()
        print('\n')
        print('EPOCH:', epoch+1)

        if model_name == 'PI-ConvTF':
            pi_data = iter(pi_add_data)

        epoch_train_losses, epoch_valid_losses = [], []
        for i, (x, y) in enumerate(train_set):
            print('EPOCH', epoch+1, '- TRAIN BATCH:', i+1)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            out = model(x)
            loss = loss_func(out, y)

            if model_name == 'PI-ConvTF':
                pi_cur = next(pi_data)[0].to(DEVICE)
                call_from_pred = BS_PDE(out, pi_cur, DEVICE)
                pi_loss_target = torch.zeros_like(call_from_pred).to(DEVICE)
                
                pi_loss = loss_func(call_from_pred, pi_loss_target)

                loss += pi_loss * model_hp['pinn_loss_weight']

            loss.backward()
            optimizer.step()

            mape_loss = torch.sum(MAPE(out, y)).item() / (out.size(dim=0)*grid*grid)
            epoch_train_losses.append(mape_loss)
        
        epoch_train_loss = sum(epoch_train_losses) / (i+1)
        train_losses.append(epoch_train_loss)
        print('TRAIN LOSS:', epoch_train_loss)

        model.eval()

        with torch.no_grad():
            for i, (x, y) in enumerate(valid_set):
                print('EPOCH', epoch+1, '- VALID BATCH:', i+1)
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = model(x)
                
                mape_loss = torch.sum(MAPE(out, y)).item() / (out.size(dim=0)*grid*grid)
                epoch_valid_losses.append(mape_loss)
        epoch_valid_loss = sum(epoch_valid_losses) / (i+1)
        valid_losses.append(epoch_valid_loss)
        print('VALID LOSS:', epoch_valid_loss)

        lr_scheduler.step(epoch_valid_loss)

        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            torch.save(model.state_dict(), '../models/models/' + model_save_name + '.pt')
    
    model_configs['timestep'] = configs.timestep
    model_configs['valid_split'] = configs.valid_split
    model_configs['m_range'] = m_range
    model_configs['ttm_range'] = ttm_range
    model_configs['grid'] = grid
    model_configs['train_start'] = configs.train_start
    model_configs['train_end'] = configs.train_end
    model_configs['test_start'] = configs.test_start
    model_configs['test_end'] = configs.test_end
    with open('../models/configs/' + model_save_name + '_configs.json') as outfile:
        json.dump(model_configs, outfile)

    return model, model_save_name, DEVICE

# Tester for ConvLSTM, SA-ConvLSTM, ConvTF, PI-ConvTF
def test(model, test_set, model_save_name, device):
    '''
    Makes volatility predictions with test set data and saves them as npy file.
    Also saves MAPE calculated with preds and y_test.
    Shape of both saved npy file is (test days * num_ttm * num_moneyness,)

    inputs
        -model: torch model that will make predictions
        -test_set: torch DataLoader object with test data
        -model_save_name: chosen file savename for model (str)
        -device: chosen device that model trained on; will be used for test (str)
    '''

    preds, test_losses = [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_set):
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.flatten())

            mape_loss = MAPE(out, y)
            test_losses.append(mape_loss)

    preds = np.array(torch.cat(preds))
    test_losses = np.array(torch.cat(test_losses))

    np.save('../results/preds/' + model_save_name + '_preds.npy', preds)
    np.save('../results/MAPEs/vol/' + model_save_name + '_mapes.npy', test_losses)