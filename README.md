# Training Environment

- tf: python 3.6.9, PINN model
- torch: python 3.7.11, ConvLSTM & SA-ConvLSTM & ConvTF & PI-ConvTF models
- OS: linux, Ubuntu 20.04
- CUDA: 11.1 (Driver Version 455.45.01)


# Makefile Structure aa

1. Setup
- COMMAND: `make setup fw=tf` OR `make setup fw=torch`
- args: fw (tf OR torch)
- recommend using different virtual environments or docker containers for tensorflow and pytorch
- installs dependencies listed in tensorflow_src/requirements.txt OR pytorch_src/requirements.txt
- For linux systems, in the case of 'Could not load libcusolver.so.11' error when trying to train tf based models, try the following: 

    ln -s /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11
    export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

2. Preprocess
- start virtual environment or docker container first
- COMMAND: `make preprocess`
- refer to 'Data' section for further details 
- args: m_range, ttm_range, grid_size
- preprocesses and saves data according to moneyness range & ttm range & grid_size 

3. Train and Test Models
- start virtual environment or docker container first
- COMMAND: `make model`
- args: m_range, ttm_range, grid_size, conv_more_inputs, fw, model_name, model_save, gpu
- Ex) `make model m_low=0.75 m_high=1.25 ttm_low=0.04 ttm_high=1 g=20 conv_more_inputs=1 fw=torch model_name=all model_save=Model1 device=gpu:0`
- if fw=tf, only the PINN model is trained and conv_more_inputs and model_name arguments are not requried
- model_name should be one of 'ConvLSTM', 'SA-ConvLSTM', 'ConvTF', 'PI-ConvTF', 'all'
- if model_name=all, trains all models specified by framework (torch); in this case conv_more_inputs=1 will train the more inputs version for ConvLSTM, SA-ConvLSTM, and ConvTF models (PI-ConvTF is not trained)
- model_save denotes the suffix of filename to save trained model by, i.e. if model_name=ConvLSTM and model_save=Model1, the trained ConvLSTM model will be saved with filename 'ConvLSTM_Model1.pt'
- for more inputs case, example of saved file name would be 'more_inputs_ConvLSTM_Model1.hdf5'
- extra requirements: dependencies (recommend using virtual environments) need to be set up for tensorflow OR pytorch, discretionary adjustment of hyperparameters in tensorflow_src/configs.py OR pytorch_src/configs.py
- trains models specified by fw OR model_name & conv_more_inputs
-> logs train and valid losses in tensorboard
-> saves models (via .pt and .hdf5 extensions) and respective hyperparameter configurations (as JSON files) in models/models/ and models/configs/
- makes volatility predictions on test set for models specified by model_name and evaluates MAPEs
-> saves predictions as numpy file in results/preds/ and MAPEs as numpy file in results/MAPEs/vol/

4. Evaluate MAPEs and Visualize
- start virtual environment or docker container first
- COMMAND: `make eval`
- args: call option price outlier exclusion boundary percentile low & high, include_more, model_save
- Ex) `make eval p_low=10 p_high=90 include_more=0 model_save=Model1`
- evaluates MAPEs for call option price from all models with saved filename suffix model_save and saves them in results/MAPEs/call/
- visualizes comparison of evaluated MAPEs for volatility and call option price across models and saves figures respectively in results/MAPEs/vol/ and results/MAPEs/call/
- if include_more is set, visualization of MAPEs will include more_inputs versions with the same model_save filename

=> Preprocessing and evaluation is recommended to take place within the tensorflow virtual environment / docker container.


# Data
The raw data for all experiments was purchased from https://datashop.cboe.com/option-eod-summary. 

## Specifications for data purchase
- Purchase Type: Historical
- Underlying Symbols: ^SPX
- Dates: as denoted in data/data_dates.txt
- Calcs: Include

=> Sample raw data file for one day is provided in data/raw/

## Preprocess Formats
Specifications include moneyness range, time to maturity (ttm) range, and grid size, which can be explicitly set in the command line. These are not required fields as they have default settings. Also, there are 3 data formats that preprocessing aims to build, which is as follows.

- pinn: pointwise data of (ttm, moneyness, volatility, underlying price, risk free rate, call option price) 
- conv: volatility data, data shape (num ttm x num moneyness) 
- pi_conv_tf: (ttm; volatility; underlying price; risk free rate; strike price) data, data shape (5 x num ttm x num moneyness)

The following command will generate all 3 data formats for all data dates according to the default moneyness range, ttm range, and grid size.

    make preprocess

To specify moneyness range, ttm range, and grid size, type custom settings as follows.

    make preprocess m_low={m_low} m_high={m_high} ttm_low={ttm_low} ttm_high={ttm_high} g={g}

Sample preprocessed data of one day is provided for each data format under data/m_0.9_1.1/ttm_0.04_1/grid_20/ in 4 subfolders: pinn/, conv/, pi_conv_tf/.