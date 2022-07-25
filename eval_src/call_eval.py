import numpy as np
import pandas as pd
import json
import datetime as dt

import os
import sys

sys.path.append('..')

from preprocess import call_price

import argparse

# Helper function for excluding outlier prediction call option price values
def reject_outliers(data, call, p_low, p_high):
    total = np.concatenate((data, call), axis=-1)

    total = total[(total[:, 0] < np.percentile(total[:, 0], p_high)) & (total[:, 0] > np.percentile(total[:, 0], p_low))]

    return total[:, 0], total[:, 1]

# Retrieves data needed for evaluation of call option price and true call values
# based on test days denoted by configs of target saved model filename
# If days_only=True, returns only test days data (not returned at all when False)
def get_eval_data(save_name, days_only=False):

    # Get test_start and test_end day
    target_configs = [c for c in os.listdir('../models/configs') if save_name == c.split('_')[-2]]
    f = open('../models/configs/' + target_configs[0])
    config_data = json.load(f)
    test_start = dt.datetime.strptime(config_data['test_start'], '%Y%m%d')
    test_end = dt.datetime.strptime(config_data['test_end'], '%Y%m%d')
    m_range, ttm_range, grid = config_data['m_range'], config_data['ttm_range'], config_data['grid']
    f.close()

    # Get ttm, s, r, C_bs, k data
    test_days = []
    src_dir = '../data/m_' + str(m_range[0]) + '_' + str(m_range[1]) + '/ttm_' + str(ttm_range[0]) + '_' + str(ttm_range[1]) + '/grid_' + str(grid) + '/pinn'
    f_names = os.listdir(src_dir)
    for f in f_names:
        day_time = dt.datetime.strptime(f[:10], '%Y-%m-%d').timestamp()

        if day_time >= test_start.timestamp() and day_time <= test_end.timestamp():
            test_days.append(f[:10])

    test_days = sorted(test_days)

    if days_only:
        return test_days

    df_list = []
    for day in test_days:
        df = pd.read_csv(src_dir + '/' + day + '.csv', index_col=0)
        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df['k'] = df['moneyness'] * df['s']

    market_data = [np.array(df['ttm']), np.array(df['s']), np.array(df['r']), np.array(df['k'])]

    true_call = np.array(df['C_bs'])

    # Get predicted volatility for every model w/ save_name
    vol_pred_names = [v for v in os.listdir('../results/preds') if save_name == v.split('_')[-2]]
    vol_preds = {}

    for v in vol_pred_names:
        vol_preds[v] = np.load('../results/preds/' + v)

    return market_data, vol_preds, true_call, grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-p', type=int, default=[10, 90], help='low and high percentile value to exclude predicted call option prices', nargs=2)
    parser.add_argument('-save', type=str, required=True, help='filename to retrieve relevant model volatility predictions for evaluation')

    args = parser.parse_args()
    
    market_data, vol_preds, true_call, grid = get_eval_data(args.save)

    # For each model
    for i in range(len(vol_preds)):
        v = list(vol_preds.keys())[i]
        
        mapes = []
        vol_data = vol_preds[v]
        # For each day
        for j in range(grid*grid - 1, vol_data.shape[0], grid*grid):
            ttm = np.expand_dims(market_data[0][j-(grid*grid-1):j+1], -1)
            s = np.expand_dims(market_data[1][j-(grid*grid-1):j+1], -1)
            r = np.expand_dims(market_data[2][j-(grid*grid-1):j+1], -1)
            k = np.expand_dims(market_data[3][j-(grid*grid-1):j+1], -1)
            vol = np.expand_dims(vol_data[j-(grid*grid-1):j+1], -1)
            call = np.expand_dims(true_call[j-(grid*grid-1):j+1], -1)

            call_preds = []

            # For each point
            for n in range(grid*grid):
                call_pred = call_price(s[n], k[n], ttm[n], r[n], vol[n])
                call_preds.append(call_pred)

            call_preds = np.array(call_preds).reshape(grid*grid, 1)

            # Reject Outliers
            call_preds, call = reject_outliers(call_preds, call, args.p[0], args.p[1])

            mapes.append(100*np.sum(np.divide(abs(call - call_preds), call)) / call.shape[0])
        
        os.makedirs('../results/MAPEs/call/', exist_ok=True)
        # Save daily MAPEs per model
        np.save('../results/MAPEs/call/' + v[:-9] + 'call_mapes.npy', np.array(mapes))
