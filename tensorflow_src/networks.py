import numpy as np
import pandas as pd
import sciann as sn
from sciann.utils.math import sign
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime as dt

from tensorflow import keras
from tensorflow.keras import layers

from os import listdir
from os.path import isfile, join

import tensorflow as tf

import json

import configs

# PINN Model Trainer and Tester 
class PINN_TrainTest:

    def __init__(self, model_configs, m_range, ttm_range, grid):
        '''
        Module for training and testing (making predictions) for the PINN Model

        inputs
            -model_configs: PINN Model hyperparameters (dict)
            -m_range: [moneyness low, moneyness high] (list)
            -ttm_range: [ttm low, ttm high] (list)
            -grid: grid size (int)
        '''

        print('\n')
        print('-GPU CHECK-')
        print(tf.config.list_physical_devices('GPU'))
        print('\n')

        self.epochs = model_configs['epochs']
        self.batch_size = model_configs['batch_size']
        self.optimizer = model_configs['optimizer']
        self.init_lr = model_configs['init_lr']
        self.reduce_lr_after = model_configs['reduce_lr_after']
        self.stop_lr_value = model_configs['stop_lr_value']
        self.model_hp = model_configs['model_hp']

        self.model_load_path = model_configs['model_load_path']
        self.model_save_name = model_configs['model_save_name']

        self.m_range = m_range
        self.ttm_range = ttm_range
        self.grid = grid

        # Model configuration to save after training and saving model
        self.model_configs = model_configs
        self.model_configs['m_range'] = self.m_range
        self.model_configs['ttm_range'] = self.ttm_range
        self.model_configs['grid'] = self.grid
        self.model_configs['train_start'] = configs.train_start
        self.model_configs['train_end'] = configs.train_end
        self.model_configs['test_start'] = configs.test_start
        self.model_configs['test_end'] = configs.test_end
    
    def train(self, x_train, y_train):
        '''
        Builds and trains ConvLSTM model according to configs.
        During training, train and valid losses are logged on tensorboard, and
        the trained model is saved.
        If conducting transfer learning, loads model from 'model_load_path',
        resumes training, and saves it to path created from 'model_save_name'.

        inputs
            -x_train: npy array (ttm, moneyness, underlying price, risk free rate) data of shape (num points x 4)
            -y_train: npy array volatility data of shape (num points x 1) 
        returns
            -model: trained model
            -vol: trained volatility function
        '''

        log_dir = 'logs/' + self.model_save_name
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Input Variables
        t = sn.Variable('t', dtype='float64')
        m = sn.Variable('m', dtype='float64')
        s = sn.Variable('s', dtype='float64')
        r = sn.Variable('r', dtype='float64')

        input_data = [
            np.expand_dims(x_train[:, 0], axis=-1),
            np.expand_dims(x_train[:, 1], axis=-1),
            np.expand_dims(x_train[:, 2], axis=-1),
            np.expand_dims(x_train[:, 3], axis=-1)
        ]

        # Target Variables (as NN)
        vol = sn.Functional('vol', [t, m, s, r], self.model_hp['volatility']['hidden_layers'], activation=self.model_hp['volatility']['activation'])
        c = sn.Functional('c', [t, m, s, r], self.model_hp['call']['hidden_layers'], activation=self.model_hp['call']['activation'])

        # Derivatives for PINN loss
        c_t = sn.diff(c, t)
        c_s = sn.diff(c, s)
        c_ss = sn.diff(c, s, order=2)

        # Objective Loss
        ## pinn_target is set s.t. the change of variable scheme
        ## that happened during rescaling is taken into account (for s)
        pinn_target = c_t - r*s*c_s + r*c - ((vol**2)/2)*(c_ss*(s**2)-c_s*s)
        vol_target = sn.Data(vol)

        data_pinn_target = 'zeros'
        data_vol_target = y_train
        target_data = [data_pinn_target, data_vol_target]

        # Set up and train model
        model = sn.SciModel(
            inputs=[t, m, s, r],
            targets=[pinn_target, vol_target],
            loss_func='mae',
            optimizer=self.optimizer
        )        

        if self.model_load_path is not None:
            model.load_weights(self.model_load_path)

        history = model.train(
            x_true=input_data,
            y_true=target_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            learning_rate=self.init_lr,
            reduce_lr_after=self.reduce_lr_after,
            verbose=1,
            stop_lr_value=self.stop_lr_value,
            callbacks=[tensorboard]
        )

        # Save model and model configs
        model.save_weights('../models/models/' + self.model_save_name + '.hdf5')
        with open('../models/configs/' + self.model_save_name + '_configs.json', 'w') as outfile:
            json.dump(self.model_configs, outfile)

        return model, vol

    def test(self, model, x_test, y_test):
        '''
        Makes volatility predictions with test set data and saves them as npy file.
        Also saves MAPE calculated with preds and y_test.
        Shape of both saved npy file is (test days * num_ttm * num_moneyness,)

        inputs
            -model: keras model that will make predictions
            -x_test: (ttm, moneyness, underlying price, risk free rate) data of shape (test days * num_ttm * num_m x 4)
            -y_test: volatility data of shape (test days * num_ttm * num_m x 1)
        '''
        preds = vol.eval(
            model,
            [
                np.expand_dims(x_test[:, 0], axis=-1),
                np.expand_dims(x_test[:, 1], axis=-1),
                np.expand_dims(x_test[:, 2], axis=-1),
                np.expand_dims(x_test[:, 3], axis=-1)
            ]
        )
        np.save('../results/preds/' + self.model_save_name + '_preds.npy', preds)

        y_test = y_test.flatten()
        mapes = np.divide(100*abs(y_test - preds), y_test)
        np.save('../results/MAPEs/vol/' + self.model_save_name + '_mapes.npy', mapes)