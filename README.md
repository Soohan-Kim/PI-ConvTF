# Physics-informed Convolutional Transformer for Predicting Volatility Surface

- This repo contains implementation code for the paper [Physics-informed Convolutional Transformer for Predicting Volatility Surface](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2294799), written by Soohan Kim, Seok-Bae Yun, Hyeong-Ohk Bae, Muhyun Lee, and Youngjoon Hong.

## Training Environment

- python: 3.7.11
- OS: linux, Ubuntu 20.04
- CUDA: 11.4 (Driver Version 470.94)


## Makefile Structure 

### 1. Setup
- COMMAND: `make setup`
- installs dependencies listed in src/requirements.txt
- For linux systems, in the case of 'Could not load libcusolver.so.11' error when trying to train tf based model PINN, try the following: 

    `ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11`
    
    `export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`


### 2. Preprocess
- COMMAND: `make preprocess`
- refer to 'Data' section for further details 
- args: m_range, ttm_range, grid_size
- preprocesses and saves data according to moneyness range & ttm range & grid_size 


### 3. Train and Test Models
- COMMAND: `make model`
- args: conv_more_inputs, model_name, model_save, gpu
- Ex) `make model conv_more_inputs=1 model_name=all model_save=Model1 device=gpu:0`
- model_name should be one of 'PINN', 'ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF', 'all'
- if model_name=all, trains all models including conv_more_inputs version for 'ConvLSTM', 'SA-ConvLSTM', and 'ConvTF'
- for training specific models, the conv_more_inputs flag will decide whether to train the original input or more input version for 'ConvLSTM', 'SA-ConvLSTM', and 'ConvTF'
- model_save denotes the suffix of filename to save trained model by, i.e. if model_name=ConvLSTM and model_save=Model1, the trained ConvLSTM model will be saved with filename 'ConvLSTM_Model1.pt'
(Omit the use of '_' in model_save)
- for more inputs case, example of saved file name would be 'more_inputs_ConvLSTM_Model1.hdf5'
- extra requirements: dependencies need to be set up, discretionary adjustment of hyperparameters in src/configs.py 
- trains models specified by model_name & conv_more_inputs

    -> logs train and valid losses in tensorboard

    -> saves models (via .pt and .hdf5 extensions) and respective hyperparameter configurations (as JSON files) in models/models/ and models/configs/
- makes volatility predictions on test set for models specified by model_name and evaluates MAPEs

    -> saves predictions as numpy file in results/preds/ and MAPEs as numpy file in results/MAPEs/vol/


### 4. Evaluate MAPEs and Visualize
- start virtual environment or docker container first
- COMMAND: `make eval`
- args: call option price outlier exclusion boundary percentile low & high, include_more, model_save
- Ex) `make eval p_low=10 p_high=90 include_more=0 model_save=Model1`
- evaluates MAPEs for call option price from all models with saved filename suffix model_save and saves them in results/MAPEs/call/
- visualizes comparison of evaluated MAPEs for volatility and call option price across models and saves figures respectively in results/MAPEs/vol/ and results/MAPEs/call/
- if include_more is set, visualization of vol MAPE (not call as NaN values occur in this case due to low model accuracies) will include more_inputs versions with the same model_save filename (more versions will be denoted by dashed lines)

=> The `make` option combines steps 3 and 4.


## Data
The raw data for all experiments was purchased from https://datashop.cboe.com/option-eod-summary. 

### Specifications for data purchase
- Purchase Type: Historical
- Underlying Symbols: ^SPX
- Dates: as denoted in data/data_dates.txt
- Calcs: Include

=> Sample raw data file for one day is provided in data/raw/

### Preprocess Formats
Specifications include moneyness range, time to maturity (ttm) range, and grid size, which can be explicitly set in the command line. These are not required fields as they have default settings. Also, there are 3 data formats that preprocessing aims to build, which is as follows.

- pinn: pointwise data of (ttm, moneyness, volatility, underlying price, risk free rate, call option price) 
- conv: volatility data, data shape (num ttm x num moneyness) 
- pi_conv_tf: (ttm; volatility; underlying price; risk free rate; strike price) data, data shape (5 x num ttm x num moneyness)

The following command will generate all 3 data formats for all data dates according to the default moneyness range, ttm range, and grid size.

    make preprocess

To specify moneyness range, ttm range, and grid size, type custom settings as follows:

    make preprocess m_low={m_low} m_high={m_high} ttm_low={ttm_low} ttm_high={ttm_high} g={g}
