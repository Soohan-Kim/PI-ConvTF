from os import listdir
from os.path import isfile, join
from datetime import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Helper function for rescaling data
def rescale(inputs):
    arr = np.expand_dims(inputs.flatten(), axis=1)
    scaler = MinMaxScaler(feature_range=(configs.scale_low, configs.scale_high))
    arr = scaler.fit_transform(arr)
    out = np.reshape(arr, inputs.shape)

    return out

# Helper function for making torch DataLoader object
def make_dataloader(x_train, y_train, x_test, y_test, batch_size, pi=False):
    valid_start = int(x_train.shape[0] * (1-configs.valid_split))

    if pi:
        add_valid = x_train[valid_start:, ...]
        add_train = x_train[:valid_start, ...]

        add_train, add_valid = torch.tensor(add_train, requires_grad=True), torch.tensor(add_valid)

        train_dataset = TensorDataset(add_train, torch.empty((add_train.size(0))))
        valid_dataset = TensorDataset(add_valid, torch.empty((add_valid.size(0))))

        train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_set = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_set, valid_set

    x_valid, y_valid = x_train[valid_start:, ...], y_train[valid_start:, ...]
    x_train, y_train = x_train[:valid_start, ...], y_train[:valid_start, ...]

    x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    x_valid, y_valid = torch.FloatTensor(x_valid), torch.FloatTensor(y_valid)
    x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset = TensorDataset(x_test, y_test)

    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_set = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_set, valid_set, test_set

# Helper function for retrieving train, test filenames (str) as two lists
def collect_files(datafiles):
    train_files, test_files = [], []
    for f in datafiles:
        day_time = dt.strptime(f[:10], '%Y-%m-%d').timestamp()

        if day_time >= configs.test_start.timestamp() and day_time <= configs.test_end.timestamp():
            test_files.append(f)
        elif day_time <= configs.train_end.timestamp() and day_time >= configs.train_start.timestamp():
            train_files.append(f)

    temp = train_files[-configs.timestep:]
    temp.extend(test_files)
    test_files = sorted(temp)

    return sorted(train_files), test_files

# Get Input Data for ConvLSTM, PINN, SA-ConvLSTM, ConvTF, PI-ConvTF
def get_data(m_range, ttm_range, grid, conv_more_inputs, fw, PINN_range=None):
    '''
    Get input data based on model type, 
    data configs, and conv_more_inputs flag.

    inputs
        -m_range: [moneyness low, moneyness high] (list of floats)
        -ttm_range: [ttm low, ttm high] (list of floats)
        -grid: grid size (int)
        -conv_more_inputs: flag for also training with more inputs (bool)
        -fw: torch or tf (str)
        -PINN_range: [start idx, end idx] (list of ints)

    returns
        (dict) w/ keys 'ConvLSTM', 'PINN', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF'
        (second elements of DataLoader of 'ConvLSTM', 'SA-ConvLSTM', 'ConvTF' are excluded if conv_more_inputs flag is not set)
    => 'PINN'
        (num_points becomes end idx - start idx)
        -x_train: (ttm, moneyness, underlying price, risk free rate) data of shape (num points x 4)
        -y_train: volatility data of shape (num points x 1) 
        -x_test: (ttm, moneyness, underlying price, risk free rate) data of shape (test days * num_ttm * num_m x 4)
        -y_test: volatility data of shape (test days * num_ttm * num_m x 1)
    => 'ConvLSTM' or'SA-ConvLSTM' or 'ConvTF'
        (one sample from DataLoader is of shape (batch_size x timesteps x 1 or 5 x num_ttm x num_m))
        -train_set, valid_set, test_set: [torch DataLoader w/ only volatility inputs, torch DataLoader w/ more inputs]
    => 'PI-ConvTF'
        PI data of shape (batch_size x 4 x num_ttm x num_m) w/ ttm, underlying price, risk free rate, and strike data 
        -train_set, valid_set: [torch DataLoader w/ only volatility inputs, torch DataLoader w/ PI data]
        -> test set not needed to make predictions for PI-ConvTF (can simply use test_set from ConvTF)
    '''

    if fw == 'torch':
        import pytorch_src.configs as configs
    else:
        import tensorflow_src.configs as configs

    data = {
        'ConvLSTM': {
            'train_set': [], 'valid_set': [], 'test_set': []
        },
        'PINN': {
            'x_train': None, 'y_train': None, 'x_test': None, 'y_test': None
        },
        'SA-ConvLSTM': {
            'train_set': [], 'valid_set': [], 'test_set': []
        },
        'ConvTF': {
            'train_set': [], 'valid_set': [], 'test_set': []
        },
        'PI-ConvTF': {
            'train_set': [], 'valid_set': []
        }
    }

    from_dir = './data/m_' + str(m_range[0]) + '_' + str(m_range[1]) + '/ttm_' + str(ttm_range[0]) + '_' + str(ttm_range[1]) + '/grid_' + str(grid)

    conv_dir = from_dir + '/conv'
    more_dir = from_dir + '/pi_conv_tf'
    datafiles = sorted([f for f in listdir(conv_dir) if isfile(join(conv_dir, f))])
    morefiles = sorted([f for f in listdir(more_dir) if isfile(join(more_dir, f))])

    ## Convolution-based models Inputs Preparation ##
    # Collect Train/Test Files
    train_files, test_files = collect_files(datafiles)

    # Retrieve Train/Test Data
    train_data, test_data = [], []
    more_train, more_test = [], []
    for f in train_files:
        day_df = pd.read_csv(join(conv_dir, f), index_col=0)
        train_data.append(day_df.to_numpy())

        day_arr = np.load(join(more_dir, f[:-3] + 'npy'))
        more_train.append(day_arr)

    for f in test_files:
        day_df = pd.read_csv(join(conv_dir, f), index_col=0)
        test_data.append(day_df.to_numpy())

        day_arr = np.load(join(more_dir, f[:-3] + 'npy'))
        more_test.append(day_arr)

    train_data = np.array(train_data, dtype=object)
    test_data = np.array(test_data, dtype=object)
    more_train = np.array(more_train, dtype=object)
    more_test = np.array(more_test, dtype=object)

    # Split Train/Test Data
    x_train, y_train = [], []
    x_test, y_test = [], []
    more_train_x = []
    more_test_x = []
    pi_tf_add = []

    for x in range(configs.timestep, len(train_data)):
        x_train.append(train_data[x - configs.timestep : x, :, :])
        y_train.append(train_data[x, :, :])

        more_train_x.append(more_train[x - configs.timestep : x, :, :, :])
        pi_tf_add.append(more_train[x, :, :, :])

    for x in range(configs.timestep, len(test_data)):
        x_test.append(test_data[x - configs.timestep : x, :, :])
        y_test.append(test_data[x, :, :])

        more_test_x.append(more_test[x - configs.timestep : x, :, :, :])

    x_train, y_train = np.asarray(x_train).astype('float32'), np.asarray(y_train).astype('float32')
    x_test, y_test = np.asarray(x_test).astype('float32'), np.asarray(y_test).astype('float32')

    temp = np.asarray(pi_tf_add).astype('float32')
    pi_tf_add = np.stack((np.expand_dims(temp[:, 0, :, :], axis=-3), temp[:, 2:, :, :]), axis=-3)

    more_train_x= np.asarray(more_train_x).astype('float32')
    more_test_x = np.asarray(more_test_x).astype('float32')

    x_train, y_train = np.expand_dims(x_train, axis=-3), np.expand_dims(y_train, axis=-3)
    x_test, y_test = np.expand_dims(x_test, axis=-3), np.expand_dims(y_test, axis=-3)

    if fw == 'torch':
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        # For torch models, construct DataLoader and also split valid set (original volatility only input data)
        cv_train, cv_valid, cv_test = make_dataloader(x_train, y_train, x_test, y_test, configs.ConvLSTM['batch_size'])
        sa_train, sa_valid, sa_test = make_dataloader(x_train, y_train, x_test, y_test, configs.SAConvLSTM['batch_size'])
        tf_train, tf_valid, tf_test = make_dataloader(x_train, y_train, x_test, y_test, configs.ConvTF['batch_size'])
        pi_train, pi_valid, _ = make_dataloader(x_train, y_train, x_test, y_test, configs.PIConvTF['batch_size'])

        # Add ConvLSTM, SA-ConvLSTM, ConvTF, PI-ConvTF Input Data (original volatility only) #
        data['ConvLSTM']['train_set'].append(cv_train)
        data['ConvLSTM']['valid_set'].append(cv_valid)
        data['ConvLSTM']['test_set'].append(cv_test)
        data['SA-ConvLSTM']['train_set'].append(sa_train)
        data['SA-ConvLSTM']['valid_set'].append(sa_valid)
        data['SA-ConvLSTM']['test_set'].append(sa_test)
        data['ConvTF']['train_set'].append(tf_train)
        data['ConvTF']['valid_set'].append(tf_valid)
        data['ConvTF']['test_set'].append(tf_test)
        data['PI-ConvTF']['train_set'].append(pi_train)
        data['PI-ConvTF']['valid_set'].append(pi_valid)
    
        # Add additional data for PI-ConvTF (to incorporate PI loss) #
        pi_add_train, pi_add_valid = make_dataloader(pi_tf_add, None, None, None, configs.PIConvTF['batch_size'], pi=True)
        data['PI-ConvTF']['train_set'].append(pi_add_train)
        data['PI-ConvTF']['valid_set'].append(pi_add_valid)

    # Add more_inputs version data for ConvLSTM, SA-ConvLSTM, ConvTF #
    if conv_more_inputs:
        # Scale underlying price and strike data sequence wise 
        for i in range(more_train_x.shape[0]):
            s_data, k_data = more_train_x[i, :, 2, :, :], more_train_x[i, :, 4, :, :]
            more_train_x[i, :, 2, :, :] = rescale(s_data)
            more_train_x[i, :, 4, :, :] = rescale(k_data)

        for i in range(more_test_x.shape[0]):
            s_data, k_data = more_test_x[i, :, 2, :, :], more_test_x[i, :, 4, :, :]
            more_test_x[i, :, 2, :, :] = rescale(s_data)
            more_test_x[i, :, 4, :, :] = rescale(k_data)

        if fw == 'torch':
            cv_more_train, cv_more_valid, cv_more_test = make_dataloader(more_train_x, more_train_y, more_test_x, more_test_y, configs.ConvLSTM['batch_size'])
            sa_more_train, sa_more_valid, sa_more_test = make_dataloader(more_train_x, more_train_y, more_test_x, more_test_y, configs.SAConvLSTM['batch_size'])
            tf_more_train, tf_more_valid, tf_more_test = make_dataloader(more_train_x, more_train_y, more_test_x, more_test_y, configs.ConvTF['batch_size'])

            data['ConvLSTM']['train_set'].append(cv_more_train)
            data['ConvLSTM']['valid_set'].append(cv_more_valid)
            data['ConvLSTM']['test_set'].append(cv_more_test)
            data['SA-ConvLSTM']['train_set'].append(sa_more_train)
            data['SA-ConvLSTM']['valid_set'].append(sa_more_valid)
            data['SA-ConvLSTM']['test_set'].append(sa_more_test)
            data['ConvTF']['train_set'].append(tf_more_train)
            data['ConvTF']['valid_set'].append(tf_more_valid)
            data['ConvTF']['test_set'].append(tf_more_test)

    ## PINN model Input Preparation ##
    pinn_dir = from_dir + '/pinn'
    datafiles = sorted([f for f in listdir(pinn_dir) if isfile(join(pinn_dir, f))])

    # Collect train, test files
    train_files, test_files = collect_files(datafiles)

    # Retrieve train, test data
    train_df_list, test_df_list = [], []
    for train in train_files:
        df = pd.read_csv(join(pinn_dir, train), index_col=0)
        train_df_list.append(df)

    for test in test_files:
        df = pd.read_csv(join(pinn_dir, test), index_col=0)
        test_df_list.append(df)

    train_df = pd.concat(train_df_list, axis=0, ignore_index=True)
    test_df = pd.concat(test_df_list, axis=0, ignore_index=True)

    # Sample data points
    train_df = train_df.iloc[PINN_range[0]: PINN_range[1], :]

    # Rescale underlying price
    s_train = pd.Series(rescale(np.array(train_df['s'])))
    train_df['s'] = s_train

    # Shape x, y data
    x_train = np.array(train_df.drop(columns=['sigma', 'C_bs']))
    y_train = np.array(train_df['sigma']).reshape(len(train_df), 1)
    x_test = np.array(test_df.drop(columns=['sigma', 'C_bs']))
    y_test = np.array(test_df['sigma']).reshape(len(test_df), 1)

    # Add PINN Input Data #
    data['PINN']['x_train'] = x_train
    data['PINN']['y_train'] = y_train
    data['PINN']['x_test'] = x_test
    data['PINN']['y_test'] = y_test

    return data
