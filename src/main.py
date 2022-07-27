import configs
import get_data
import utils

import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-cmi', type=int, default=0, choices=[0, 1], help='option for training ConvLSTM, SA-ConvLSTM, ConvTF with more inputs; set (1) or not (0)')
    parser.add_argument(
        '-model', type=str, choices=['PINN', 'ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF', 'all'],
        help='model type to train; if all trains all models within current framework'
    )
    parser.add_argument(
        '-save', type=str, required=True,
        help='filename suffix to save trained models by'
    )
    parser.add_argument('-device', type=str, default='gpu:0', help='device to train model on')

    args = parser.parse_args()
    
    if args.model == 'all':
        args.cmi = 1

    models = ['ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF']
    model_configs = [configs.ConvLSTM, configs.SAConvLSTM, configs.ConvTF, configs.PIConvTF]

    # Suffixing model_save_name
    for i in range(4):
        model_configs[i]['model_save_name'] = models[i] + '_' + args.save
    configs.PINN['model_save_name'] = 'PINN_' + args.save

    # Get Relevant Train/Valid/Test Data
    data = get_data.get_data(
        m_range=configs.m, ttm_range=configs.ttm, grid=configs.g, conv_more_inputs=args.cmi,
        PINN_range=configs.PINN['PINN_range']
    )

    # Torch models Train & Test
    if args.model != 'PINN':
        for i in range(4):
            # If model is specified, train only that model
            if args.model in models and args.model != models[i]:
                continue
            
            # For original and more inputs case
            for j in range(2):        
                # For PI-ConvTF, only train once
                if i == 3:
                    if j == 0:
                        continue
                    idx, cmi = 0, 0
                # When training all models, train for both original and more inputs case
                elif args.model == 'all':
                    cmi = j
                    idx = j
                # If model specified, train once based on conv_more_inputs flag
                else:
                    if j == 0:
                        continue
                    cmi = args.cmi
                    idx = args.cmi
                
                pi_add = None if i < 3 else data['PI-ConvTF']['train_set'][1]
                model, model_save, device = utils.train(
                    model_name=models[i], 
                    model_configs=model_configs[i],
                    m_range=configs.m, ttm_range=configs.ttm, grid=configs.g, 
                    conv_more_inputs=cmi, 
                    data=[data[models[i]]['train_set'][idx], data[models[i]]['valid_set'][idx]],
                    device=args.device,
                    pi_add_data=pi_add
                )

                k = i if i < 3 else i-1
                utils.test(
                    model=model, test_set=data[models[k]]['test_set'][idx], model_save_name=model_save, device=device
                )

    # Tf model Train & Test
    if args.model == 'all' or args.model == 'PINN':
        if args.device[:-2] == 'gpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

        PINN_module = utils.PINN_TrainTest(
            model_configs=configs.PINN, m_range=configs.m, ttm_range=configs.ttm, grid=configs.g
        )
        PINN_model, PINN_vol = PINN_module.train(
            x_train=data['PINN']['x_train'], y_train=data['PINN']['y_train']
        )
        PINN_module.test(
            model=PINN_model, vol=PINN_vol, x_test=data['PINN']['x_test'], y_test=data['PINN']['y_test']
        )
        
    print('ALL TRAINING CYCLES COMPLETED')
