import sys

sys.path.append('..')

import configs
import get_data
import utils

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', type=float, default=[0.9, 1.1], help='moneyness range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-ttm', type=float, default=[0.04, 1], help='ttm range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-g', type=int, default=20, help='grid size (int); number of points to slice moneyness and ttm range into')
    parser.add_argument('-cmi', type=int, default=0, choices=[0, 1], help='option for training ConvLSTM, SA-ConvLSTM, ConvTF with more inputs; set (1) or not (0)')
    parser.add_argument(
        '-model', type=str, choices=['ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF', 'all'],
        help='model type to train; if all trains all models within current framework'
    )
    parser.add_argument(
        '-save', type=str, 
        help='filename suffix to save trained models by'
    )
    parser.add_argument('-device', type=str, default='gpu:0', help='device to train model on')

    args = parser.parse_args()

    models = ['ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF']
    model_configs = [configs.ConvLSTM, configs.SAConvLSTM, configs.ConvTF, configs.PIConvTF]

    # Suffixing model_save_name
    for i in range(4):
        model_configs[i]['model_save_name'] = models[i] + '_' + args.save

    # Get Relevant Train/Valid/Test Data
    data = get_data.get_data(
        m_range=args.m, ttm_range=args.ttm, grid=args.g, conv_more_inputs=args.cmi, fw='torch'
    )

    # Train & Test
    for i in range(4):
        if args.model is in models and args.model != models[i]:
            continue

        idx = args.cmi if i < 3 else 0
        pi_add = None if i < 3 else data['PI-ConvTF']['train_set'][1]
        model, model_save, device = utils.train(
            model=models[i], 
            model_configs=model_configs[i],
            m_range=args.m, ttm_range=args.ttm, grid=args.g, 
            conv_more_inputs=args.cmi, 
            data=[data[models[i]]['train_set'][idx], data[models[i]]['valid_set'][idx]],
            device=args.device,
            pi_add_data = pi_add
        )

        k = i if i < 3 else i-1
        utils.test(
            model=models[i], test_set=data[models[k]]['test_set'][idx], model_save_name=model_save, device=device
        )
