import argparse
import numpy as np
from numpy import *
import pandas as pd
from scipy import interpolate

import os
from os import listdir
from os.path import isfile, join
import yfinance as yf
import scipy.stats as si

import datetime as dt

# Classical Black-Scholes PDE Solution
def call_price(s, k, ttm, r, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = (np.log(s / k) + (r - 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))

    call = (s * si.norm.cdf(d1, 0.0, 1.0) - k * np.exp(-r * ttm) * si.norm.cdf(d2, 0.0, 1.0))

    return call

# Preprocess Module for making 'conv' and 'pinn' data
class DataPreprocess:

    def __init__(self, from_dir, to_dir_conv, to_dir_pinn, moneyness_min=0.9, moneyness_max=1.1, ttm_min=0.04, ttm_max=1, n_axis_points=20):
        file_names = sorted([f for f in listdir(from_dir) if isfile(join(from_dir, f))])
        self.from_dir = from_dir
        self.to_dir_conv = to_dir_conv
        self.to_dir_pinn = to_dir_pinn
        self.moneyness_min, self.moneyness_max = moneyness_min, moneyness_max
        self.ttm_min, self.ttm_max = ttm_min, ttm_max
        self.n_axis_points = n_axis_points

        self.r_data = yf.Ticker('^TNX').history(start=dt.datetime(2004, 1, 5), end=dt.datetime(2021, 8, 14))['Close']

        self.r_daycount = 0
        for f in file_names:
            print(f)
            self.df_new = self.get_points(f)

            X, Y, Z = self.interpolate_surface(self.df_new)

            self.save_discretized_data_for_conv(Z, f)

            self.save_close_discrete_data_for_PINN(X, Y, Z, f)

    def get_points(self, f):
        df = pd.read_csv(self.from_dir + '/' + f)

        df = df[(df['option_type'] == 'C') & (df['trade_volume'] != 0)]

        df['quote_date'] = pd.to_datetime(df['quote_date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['ttm'] = (df['expiration'] - df['quote_date']).dt.days / 365
        df['moneyness'] = df['strike'] / df['active_underlying_price_1545']

        if f[-14:-4] == self.r_data.index[self.r_daycount].strftime("%Y-%m-%d"):
            r = self.r_data[self.r_daycount]
            self.r_daycount += 1
        else:
            r = self.r_data[self.r_daycount-1]
        df['r'] = r * 0.01

        df_new = df[['ttm', 'moneyness', 'implied_volatility_1545', 'active_underlying_price_1545', 'r']]
        df_new = df_new.rename(columns={'implied_volatility_1545':'sigma', 'active_underlying_price_1545':'s'})
        df_new = df_new[
            (df_new['ttm'] >= self.ttm_min) & (df_new['ttm'] <= self.ttm_max) & (df_new['moneyness'] >= self.moneyness_min) & (
                        df_new['moneyness'] <= self.moneyness_max)]
        df_new = df_new.reset_index(drop=True)
        self.s, self.r = df_new['s'][0], df_new['r'][0]

        return df_new

    def interpolate_surface(self, df_new):
        moneyness = df_new['moneyness'].tolist()
        ttm = df_new['ttm'].tolist()
        iv = df_new['sigma'].tolist()

        X, Y = meshgrid(linspace(min(moneyness), max(moneyness), self.n_axis_points), linspace(min(ttm), max(ttm), self.n_axis_points))

        ## iv data interp ##
        Z = interpolate.griddata(array([moneyness, ttm]).T, array(iv), (X, Y), method='cubic')

        # print(Z.shape)
        # Z.shape = (# ttm, # moneyness)

        return X, Y, Z

    def save_discretized_data_for_conv(self, iv_vals, f):
        final_df = pd.DataFrame(iv_vals)
        final_df = final_df.ffill(axis=0).ffill(axis=1).bfill(axis=0).bfill(axis=1)
        final_df.to_csv(self.to_dir_conv + '/' + f[-14:-4] + '.csv')

    def save_close_discrete_data_for_PINN(self, X, Y, Z, f):

        moneyness_df, ttm_df = pd.DataFrame(X), pd.DataFrame(Y)
        sigma_df = pd.DataFrame(Z).ffill(axis=0).ffill(axis=1).bfill(axis=0).bfill(axis=1)

        total_list = []
        for i in range(len(sigma_df.index)):
            for j in range(len(sigma_df.columns)):
                c = call_price(self.s, self.s * moneyness_df.iloc[i, j], ttm_df.iloc[i, j], self.r, sigma_df.iloc[i, j])
                total_list.append([ttm_df.iloc[i, j], moneyness_df.iloc[i, j], sigma_df.iloc[i, j], self.s, self.r, c])

        total_arr = np.array(total_list)
        total_df = pd.DataFrame(total_arr, columns=['ttm', 'moneyness', 'sigma', 's', 'r', 'C_bs'])
        # print(total_df)

        total_df.to_csv(self.to_dir_pinn + '/' + f[-14:-4] + '.csv')

# Make 'pi_conv_tf' data
def make_pi_conv_tf_data(from_dir, to_dir, g):
    filelist = sorted([f for f in listdir(from_dir) if isfile(join(from_dir, f))])

    for i in range(len(filelist)):
        f = filelist[i]

        df = pd.read_csv(from_dir + '/' + f, index_col=0).drop(columns=['C_bs'])

        df['k'] = df['s'] * df['moneyness']
        df = df.drop(columns=['moneyness'])

        arr_list = []
        for col in df.columns:
            arr = df[col].to_numpy()
            arr = np.reshape(arr, (g, g))
            arr_list.append(arr)

        new_data = np.stack(arr_list, axis=0)
        np.save(to_dir + '/' + f[:-4], new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', type=float, default=[0.9, 1.1], help='moneyness range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-ttm', type=float, default=[0.04, 1], help='ttm range, requires 2 floats (low and high)', nargs=2)
    parser.add_argument('-g', type=int, default=20, help='grid size (int); number of points to slice moneyness and ttm range into')

    args = parser.parse_args()
    # print(args.m)
    # print(args.ttm)
    # print(args.g)
    m_low, m_high = args.m[0], args.m[1]
    ttm_low, ttm_high = args.ttm[0], args.ttm[1]
    g = args.g

    raw_dir = './data/raw'

    to_dir_path = './data/m_' + str(m_low) + '_' + str(m_high) + '/ttm_' + str(ttm_low) + '_' + str(ttm_high) + '/grid_' + str(g)

    conv_dir = to_dir_path + '/conv'
    os.makedirs(conv_dir, exist_ok=True)
    pinn_dir = to_dir_path + '/pinn'
    os.makedirs(pinn_dir, exist_ok=True)

    pi_conv_dir = to_dir_path + '/pi_conv_tf'
    os.makedirs(pi_conv_dir, exist_ok=True)

    conv_pinn_prep = DataPreprocess(raw_dir, conv_dir, pinn_dir, m_low, m_high, ttm_low, ttm_high, g)

    make_pi_conv_tf_data(pinn_dir, pi_conv_dir, g)