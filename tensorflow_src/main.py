import sys

sys.path.append('..')

import configs
import get_data
import networks

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', type=float, default=[0.9, 1.1], help='moneyness range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-ttm', type=float, default=[0.04, 1.0], help='ttm range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-g', type=int, default=20, help='grid size (int); number of points to slice moneyness and ttm range into')
    parser.add_argument(
        '-save', type=str, 
        help='filename suffix to save trained models by'
    )
    parser.add_argument('-device', type=str, default='gpu:0', help='device to train model on')

    args = parser.parse_args()

    configs.PINN['model_save_name'] = 'PINN_' + args.save

    if args.device[:-2] == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

    data = get_data.get_data(
        m_range=args.m, ttm_range=args.ttm, grid=args.g, conv_more_inputs=0, fw='tf', 
        PINN_range=configs.PINN['PINN_range']
    )

    PINN_module = networks.PINN_TrainTest(
        model_configs=configs.PINN, m_range=args.m, ttm_range=args.ttm, grid=args.g
    )
    PINN_model = PINN_module.train(
        x_train=data['PINN']['x_train'], y_train=data['PINN']['y_train']
    )
    PINN_module.test(
        model=PINN_model, x_test=data['PINN']['x_test'], y_test=data['PINN']['y_test']
    )
