from call_eval import get_eval_data
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

from scipy.signal import savgol_filter

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-more', type=int, default=0, help='whether to plot conv_more_inputs version together')
    parser.add_argument('-save', type=str, required=True, help='filename to retrieve relevant model volatility predictions for evaluation')

    args = parser.parse_args()

    test_days = pd.Series(np.array(get_eval_data(args.save, days_only=True)))
    test_days = test_days.apply(mpl_dates.date2num)
    date_format = mpl_dates.DateFormatter('%Y%m%d')

    fig, ax = plt.subplots(figsize=(7,4))

    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    vol_mape_files = [vf for vf in os.listdir('../results/MAPEs/vol/') if args.save == vf.split('_')[-2]]
    call_mape_files = [cf for cf in os.listdir('../results/MAPEs/call/') if args.save == cf.split('_')[-3]]

    colors = {
        'PINN': 'blue',
        'ConvLSTM': 'orange',
        'SA-ConvLSTM': 'green',
        'ConvTF': 'red',
        'PI-ConvTF': 'darkorchid'
    }

    # Vol pred mape plot (per day)
    for vf in vol_mape_files:
        vm = np.load('../results/MAPEs/vol/' + vf).reshape(len(test_days), -1)
        vm = np.mean(vm, axis=-1)

        if len(test_days) > 250:
            vm = savgol_filter(vm, 51, 3)

        model_type = vf.split('_')[-3]
        linestyle = 'dashed' if vf.split('_')[0] == 'more' else 'solid'
        if linestyle == 'dashed' and args.more == 0:
            continue
        ax.plot(test_days, vm, label=model_type, linewidth=2, color=colors[model_type], linestyle=linestyle)

    ax.set_xlabel('TEST DATE (YYYYMMDD)', fontsize=10)
    ax.set_ylabel('MAPE (%)', fontsize=10)
    plt.legend(loc='best', prop={'size':10})
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.savefig('../results/MAPEs/vol/' + args.save + '_vol_mapes.png')

    plt.close('all')

    # Call pred mape plot (per day)
    fig, ax = plt.subplots(figsize=(7,4))

    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()

    for cf in call_mape_files:
        cm = np.load('../results/MAPEs/call/' + cf)

        if len(test_days) > 250:
            cm = savgol_filter(cm, 51, 3)

        model_type = cf.split('_')[-4]
        linestyle = 'dashed' if cf.split('_')[0] == 'more' else 'solid'
        if linestyle == 'dashed' and args.more == 0:
            continue
        ax.plot(test_days, cm, label=model_type, linewidth=2, color=colors[model_type], linestyle=linestyle)

    ax.set_xlabel('TEST DATE (YYYYMMDD)', fontsize=10)
    ax.set_ylabel('MAPE (%)', fontsize=10)
    plt.legend(loc='best', prop={'size':10})
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.savefig('../results/MAPEs/call/' + args.save + '_call_mapes.png')