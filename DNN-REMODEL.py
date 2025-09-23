import sys

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import keras_tuner as kt
# from keras import backend as K
# print(tf.__version__)
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import json

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'output')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'ForestTypevalid_climate_driver_data_randomsamples.csv')

import logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(SAVE_PATH, 'DNN2model_test_log.txt'))


def make_save_path(new_save_path):
    global SAVE_PATH
    SAVE_PATH = new_save_path
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)


###
# Del Grosso, S., Parton, W., Stohlgren, T., Zheng, D., Bachelet, D., Prince, S., Hibbard, K. and Olson, R., 2008.
# Global potential net primary production predicted from vegetation class, precipitation, and temperature.
# Ecology, 89(8), pp.2117-2126.###
def cal_TNPP(prcp, tmean, verbose=False):
    f_MAP = 0.551 * prcp ** 1.055 / np.exp(0.000306 * prcp) # F(MAP) = 0.551 × MAP1.055/exp(0.000306 × MAP)
    f_MAT = 2540 / np.asarray([1 + np.exp(1.584 - 0.0622 * tmean)]) # F(MAT) = 2540/[1 + exp(1.584 − 0.0622 × MAT)]
    f_MAP = f_MAP.flatten()#.round(2)
    f_MAT = f_MAT.flatten()#.round(2)
    if verbose:
        return np.minimum(f_MAP, f_MAT), [f_MAP * (f_MAP < f_MAT), f_MAT * (f_MAP >= f_MAT)]
    return np.minimum(f_MAP, f_MAT)


def make_simudata_NPP(file_path, nx=0.0, ny=0.0, noise_x='all'): # nx, ny: noixe leve of x and y; noise_x: list, 'all'
    df = pd.read_csv(file_path)

    if 'Forest_type' in df.columns:
        df = df.drop(['Forest_type'], axis=1)

    if 'SDV_100cm' in df.columns:
        df = df.drop(["SDV_100cm"], axis=1)
    if 'Regrowth_time' in df.columns:
        df = df.drop(["Regrowth_time"], axis=1)
    if 'x' in df.columns:
        df = df.drop(["x"], axis=1)
    if 'y' in df.columns:
        df = df.drop(["y"], axis=1)

    df = df[df['prcp'] > 0] # make sure precipitation is greater than 0
    df = df[df['tmean'] > 0] # make sure tmean is greater than 0
    df_y = cal_TNPP(df['prcp'].values, df['tmean'].values)
    ## add random noise to x variables
    if nx > 0:
        for (columnName, columnData) in df.iteritems():
            if (noise_x == 'all') or (columnName in noise_x):
                print('Column Name, noise level: ', columnName, nx)
                raw_x = columnData.values
                df[columnName] = np.random.normal(loc=0, scale=nx * np.std(raw_x), size=raw_x.shape) + raw_x

    df['y'] = df_y
    ## add random noise to y variables
    if ny > 0:
        raw_y = df_y
        df['y'] = np.random.normal(loc=0, scale=ny * np.std(raw_y), size=raw_y.shape) + raw_y

    var_names = df.columns.values.tolist()

    return df, var_names


def preview_data(dataset):
    print('dataset.describe(): ', dataset.describe())
    print('Correlation: ', dataset.corr(method='pearson'))


def preprocessing_simudata(dataset, pca=True, standerize=True, train_portion=0.8):
    """Preprocessing the simulated data

    Args:
        dataset (array): orginal data
        pca (bool, optional): If conduct PCA (Principal component analysis) transformation. Defaults to True.
        standerize (bool, optional): If standerize X variables. Defaults to True.
        train_portion (float, optional): split training test data. Defaults to 0.8.

    Returns:
        train_x (array): Preprocessed X training data
        train_y (array): Preprocessed Y training data
        test_x (array): Preprocessed X test data
        test_y (array): Preprocessed Y test data
        scaler (object): The standardization scaler (see sklearn.preprocessing.StandardScaler)
        pca_model (object): The PCA object (see sklearn.decomposition.PCA)
    """
    # preview_simudata(dataset)
    # Make sure the last column is y
    dataset = dataset.reset_index(drop=True)
    shuf_idx = dataset.sample(frac=1).index  # Shuffle index of dataframe
    print('idx:', np.min(shuf_idx), np.max(shuf_idx), len(dataset))
    full_feature = dataset.iloc[shuf_idx]  # Shuffle dataframe
    np.save(os.path.join(SAVE_PATH, 'simu_shuf_idx.npy'), shuf_idx)
    # full_feature = dataset
    n_training = int(len(full_feature) * train_portion)

    if standerize:
        x_stack = full_feature.iloc[:, :-1].values.copy()  # returns a numpy array
        scaler = preprocessing.StandardScaler()
        x_scaled = scaler.fit_transform(x_stack)
        full_feature.iloc[:, :-1] = x_scaled

    else:
        scaler = None

    if pca:
        x_stack = full_feature.iloc[:, :-1].values.copy() # need to make copy
        print('x_stack.shape=', x_stack.shape)
        pca_model = PCA()  # n_components=2
        x_pca = pca_model.fit_transform(x_stack)
        full_feature.iloc[:, :-1] = x_pca
        print('x_pca.shape=', x_pca.shape)

    dataset_x = full_feature.iloc[:, :-1].values.copy()
    dataset_y = full_feature.iloc[:, -1].values.copy()
    train_x = dataset_x[:n_training, :].copy()
    train_y = dataset_y[:n_training].copy()
    # # print(train_y.head)

    test_x = dataset_x[n_training:, :].copy()
    test_y = dataset_y[n_training:].copy()
    if pca:
        return train_x, train_y, test_x, test_y, scaler, pca_model
    return train_x, train_y, test_x, test_y, scaler


def opt_DNN(hp): # Create DNN model for hyperband tune
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(5,)))

    for i in range(hp.Int("num_layers", 2, 20)):
        model.add(
            layers.Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=1024, step=32),
                activation="relu",
            )
        )

    model.add(tf.keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error')

    return model


def tune_DNN_dataset(train_x, train_y, test_x, test_y, nx=0, ny=0):
    """Tune model parameters

    Args:
        train_x (array): X input for training
        train_y (array): Y input for training
        test_x (array): X input for test
        test_y (array): Y input for test
        nx (int, optional): X noise level for output folder labeling. Defaults to 0.
        ny (int, optional): Y noise level for output folder labeling. Defaults to 0.

    Returns:
        object: optimzied model
    """
    tuner = kt.Hyperband(opt_DNN,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                         directory=SAVE_PATH,
                         project_name="TNPP_hyperband_nx{}_ny{}".format(nx, ny),)
    print('tuner.search_space_summary()', tuner.search_space_summary())
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(train_x, train_y,
                 epochs=10,
                 # batch_size=64,
                 validation_data=(test_x, test_y),
                 use_multiprocessing=True,
                 workers=40,
                 callbacks=[stop_early])
    # print('Summary')
    # print(tuner.results_summary())
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print('best hyperparameters')
    print(best_hps.values)
    print('best hyperparameters summary')
    best_models = tuner.get_best_models()
    print(best_models[0].summary())
    return tuner


def make_DNN_simu_opt_json(json_path):
    """Initialize model architecture based on previously tune results

    Args:
        json_path (_type_): tune model architecture

    Returns:
        object: Initialize model
    """
    with open(json_path, 'r') as f:
        model_json = json.load(f)
        model_json_dic = json.loads(model_json)
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(5,)))
    for key in model_json_dic['config']['layers']:
        if 'units' in key['config'].keys():
            model.add(layers.Dense(int(key['config']['units']), activation='relu'))
    return model


def compile_fit_DNN_simu(model, train_x, train_y, return_history=False):
    """compile and fit DNN model

    Args:
        model (object): DNN model with preset architecture
        train_x (array): X input
        train_y (array): Y input
        return_history (bool, optional): If return model fit history. Defaults to False.

    Returns:
        object: fitted model
    """
    model.compile(loss='mean_squared_error',#'mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    history = model.fit(
        train_x, train_y,
        validation_split=0.5, verbose=0,
        epochs=30)  # verbose=0,

    if return_history:
        return model, history
    return model


def get_relu_par(model):
    """returns a list of activation function. The function takes X data record as input and returns matrix of neuron results (zero or the linear transofrmation result)

    Args:
        model (object): DNN model

    Returns:
        list: list of activation function
    """
    get_relu_outputs = []
    for i in range(len(model.layers)):
        get_relu_outputs.append(K.function([model.layers[0].input], [model.layers[i].output]))
    return get_relu_outputs


def cal_x_sep_mp(x, model_path, return_intercept=True):
    """This is the key function that calcualte variable contribution from DNN model parameters

    Args:
        x (arr): one X data record
        model_path (str): The folder directory where the model architecture file saved.
        return_intercept (bool, optional): If returnt the DNN model bias (intercepts of linear transformation). Defaults to True.

    Returns:
        _type_: _description_
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    ## Restore the DNN model. Python cannot pass DNN model as function parameter, so I have to restore the model. ##
    model_arch_path = os.path.join(os.path.dirname(model_path), 'simu_model_architecture')
    model = make_DNN_simu_opt_json(model_arch_path)
    model.load_weights(model_path).expect_partial()
    
    weights = model.weights # load DNN parameters
    get_relu_outputs = get_relu_par(model) # load activation function list. The function takes X data record as input and returns matrix of neuron results (zero or the linear transofrmation result)
    n_layer = len(weights) // 2 # Each hidden layer has two layer of weights: variable coefficients and bias
    x_est = x.copy().squeeze()
    rec = [] # a list to store variable contribution and bias
    for i_coef in range(x_est.shape[0]): # for each variable
        x_i = x_est[i_coef]
        for i_layer in range(0, len(weights), 2): # for each hidden layer
            if i_layer == 0:
                # print('first layer')
                x_i = np.expand_dims(x_i * weights[0].numpy().squeeze()[i_coef, :], axis=0) # the first layer is a linear transformation of X data record (essentially 1-d array)
            elif i_layer // 2 < n_layer - 1:
                # print('mid layers')
                x_i = np.matmul(x_i, weights[i_layer].numpy().squeeze()) # middle years need to calculate the cumulated variable contribution from previous hidden layers
            else:
                # print('last layer')
                x_i = x_i * weights[i_layer].numpy().squeeze() # The last layer only have one neuron as final prediction
            relu = np.asarray(get_relu_outputs[i_layer // 2](x)).squeeze() # hidden layer mask derived from ReLU activation function
            x_i = x_i * (np.asarray(relu) > 0) # apply the hidden layer mask
        rec.append(np.sum(x_i.squeeze())) # store variable contribution
    if not return_intercept:
        return rec
    else: # calcualte bias
        x_i = 1
        for i_layer in range(0, len(weights), 2): # for each hidden layer
            if i_layer == 0:
                # print('first layer')
                x_i = np.expand_dims(x_i * weights[1].numpy().squeeze(), axis=0)
            elif i_layer // 2 < n_layer - 1:
                # print('mid layers')
                x_i = np.matmul(x_i, weights[i_layer].numpy().squeeze()) + weights[i_layer + 1].numpy().squeeze()
            else:
                # print('last layer')
                x_i = x_i * weights[i_layer].numpy().squeeze()
            relu = np.asarray(get_relu_outputs[i_layer // 2](x)).squeeze()
            x_i = x_i * (np.asarray(relu) > 0)
        rec.append(np.sum(x_i.squeeze()) + weights[-1].numpy().squeeze()) # store bias
        return rec


def get_var_contrb_mp(fea_x, model_path, ncpu=2):
    """Calculate the variable contribution of input X (after preprocessing)

    Args:
        fea_x (array): input X
        model_path (str): The folder directory where the model architecture file saved.
        ncpu (int, optional): number of cpus for parallel processesing. Defaults to 2.

    Returns:
        array: the variable contribution of input X
    """
    func = partial(cal_x_sep_mp,
                   model_path=model_path)

    # arr_subs = np.array_split(fea_x, fea_x.shape[0], axis=0)
    # print(len(arr_subs))
    pool = Pool(ncpu)
    rec = pool.map(func, fea_x)

    contrb_arr = np.vstack(rec)
    return contrb_arr


def get_var_contr_raw(fea_x, coef_arr, scaler=None, pca=None, return_bias=False):
    """Convert the variable contribution of input X (after preprocessing) to contribution of original X (before preprocessing).

    Args:
        fea_x (array): input X
        coef_arr (array): contribution of input X
        scaler (object, optional): The standardization scaler (see sklearn.preprocessing.StandardScaler). Defaults to None.
        pca (object, optional): The PCA object (see sklearn.decomposition.PCA). Defaults to None.
        return_bias (bool, optional): If returnt the DNN model bias (intercepts of linear transformation). Defaults to False.

    Returns:
        array: contribution of original X
    """
    slope_arr = coef_arr[:, :-1] / (fea_x + sys.float_info.epsilon) # calcualte the slope for input X variables
    if pca is not None:
        slope_arr_raw = np.dot(pca.components_.T, slope_arr.T).T # reverse the pca process
    else:
        slope_arr_raw = slope_arr.copy()

    raw_x = fea_x.copy()
    if pca is not None:
        raw_x = pca.inverse_transform(fea_x)
    if scaler is not None: # reverse the standardization
        raw_x = scaler.inverse_transform(raw_x)
        if return_bias:
            return raw_x * slope_arr_raw / scaler.scale_ - scaler.mean_ / scaler.scale_ * slope_arr_raw, coef_arr[:, -1]
        else:
            return raw_x * slope_arr_raw / scaler.scale_ - scaler.mean_ / scaler.scale_ * slope_arr_raw
    if return_bias:
        return raw_x * slope_arr_raw, coef_arr[:, -1]
    else:
        return raw_x * slope_arr_raw  # - scaler.mean_ / scaler.scale_ * slope_arr_raw


def get_var_slopes_raw(fea_x, coef_arr, scaler=None, pca=None):
    """Calculate variable slope for orginal X variables. 

    Args:
        fea_x (array): input X variables (after preprocessing)
        coef_arr (array): contribution for input X variables
        scaler (object, optional): The standardization scaler (see sklearn.preprocessing.StandardScaler). Defaults to None.
        pca (object, optional): The PCA object (see sklearn.decomposition.PCA). Defaults to None.

    Returns:
        array: variable slope
    """
    slope_arr = coef_arr[:, :-1] / (fea_x + sys.float_info.epsilon) # calcualte the slope for input X variables
    if pca is not None:
        slope_arr_raw = np.dot(pca.components_.T, slope_arr.T).T # reverse the pca process
    else:
        slope_arr_raw = slope_arr.copy()
    if scaler is not None:
      slope_arr_raw = slope_arr_raw / scaler.scale_ # reverse the standardization scale
    return slope_arr_raw


def plt_contr_var(fea_x, contr_arr, var_names, plt_type='vector', save_fig='contr_var.png'):
    """Plot variable contributions (y-axis) vs variables (x-axis)

    Args:
        fea_x (array): X variables
        contr_arr (array): X variable contributions
        var_names (list): list of variable names for axis labeling
        plt_type (str, optional): 'vector' will plot variable contribution with the same variable. 'matrix' will plot variable contribution with all variables. the matrix may be useful to examine the variable contribution as a function of other variables. Defaults to 'vector'.
        save_fig (str, optional): output figure name. Defaults to 'contr_var.png'.
    """
    nrow = fea_x.shape[1]
    ncol = contr_arr.shape[1]
    if 'bias' not in var_names:
        var_names.append('bias')

    if plt_type == 'matrix':
        fig, axs = plt.subplots(nrow, ncol, figsize=(20, 20))
        # fig = plt.figure(figsize=(20, 20))
        i_plt = 1
        for i in range(nrow):
            for j in range(ncol):
                # Calculate the point density
                xy = np.vstack([fea_x[:, i], contr_arr[:, j]])
                z = gaussian_kde(xy)(xy)
                axs[i, j].scatter(fea_x[:, i], contr_arr[:, j], c=z)
                axs[i, j].set_xlabel(var_names[i])
                axs[i, j].set_ylabel(var_names[j])

                axs[i, j].set_xlabel(var_names[i])
                axs[i, j].set_ylabel(var_names[j] + ' contribution')
                i_plt += 1

        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(os.path.join(SAVE_PATH, save_fig), dpi=300)
        else:
            plt.show()
    elif plt_type == 'vector':
        # fig, axs = plt.subplots(nrow, ncol, figsize=(20, 20), projection='scatter_density')
        n_sub = np.min([nrow, ncol])
        fig = plt.figure(figsize=(4*n_sub, 4))
        i_plt = 1
        clean_x = pd.read_csv(r'H:\qzhou\Product_analysis\ForestTypevalid_climate_driver_data_randomsamples.csv')
        _, pred_verbose = cal_TNPP(clean_x['prcp'].values, clean_x['tmean'].values, verbose=True)
        # _, pred_verbose = cal_TNPP(fea_x[:, 0], fea_x[:, 1], verbose=True) # fea_x[:, 0] is prcp, fea_x[:, 1] is Tmean
        for i in range(n_sub):
            # Calculate the point density
            ax = fig.add_subplot(1, n_sub, i_plt)
            plt_idx = contr_arr[:, i] != None
            ## data density
            xy = np.vstack([fea_x[plt_idx, i], contr_arr[plt_idx, i]])
            z = gaussian_kde(xy)(xy)

            ax.scatter(fea_x[plt_idx, i], contr_arr[plt_idx, i], c=z) #z[plt_idx]
            ax.set_xlabel(var_names[i])
            ax.set_ylabel(var_names[i] + ' contribution')

            if i < len(pred_verbose):
                ax.scatter(clean_x[var_names[i]][plt_idx], pred_verbose[i][plt_idx], s=0.5)

            i_plt += 1

        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(os.path.join(SAVE_PATH, save_fig), dpi=400)
        else:
            plt.show()
    else:
        print('plot type is either vector or matrix.')


def plt_slope_var(fea_x, slope_arr, var_names, plt_type='vector', save_fig='slope_var_matrix.png'):
    """Plot variable slopes (y-axis) vs variables (x-axis)

    Args:
        fea_x (_type_): X variables
        slope_arr (_type_): X variable slopes
        var_names (_type_): list of variable names
        plt_type (str, optional): 'vector' will plot variable slope with the same variable. 'matrix' will plot variable slope with all variables. the matrix may be useful to examine the variable slope as a function of other variables. Defaults to 'vector'.
        save_fig (str, optional): output figure name. Defaults to 'slope_var_matrix.png'.
    """
    nrow = fea_x.shape[1]
    ncol = slope_arr.shape[1]
    delta_x = 1
    if 'bias' not in var_names:
        var_names.append('bias')
    print('var_names=', var_names)
    if plt_type == 'matrix':
        fig, axs = plt.subplots(nrow, ncol, figsize=(20, 20))
        i_plt = 1
        for i in range(nrow):
            for j in range(ncol):
                # Calculate the point density
                xy = np.vstack([fea_x[:, i], slope_arr[:, j]])
                z = gaussian_kde(xy)(xy)
                axs[i, j].scatter(fea_x[:, i], slope_arr[:, j], c=z)
                axs[i, j].set_xlabel(var_names[i])
                axs[i, j].set_ylabel(var_names[j])

                axs[i, j].set_xlabel(var_names[i])
                axs[i, j].set_ylabel(var_names[j])
                i_plt += 1

        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(os.path.join(SAVE_PATH, save_fig), dpi=300)
        else:
            plt.show()
    elif plt_type == 'vector':
        n_sub = np.min([nrow, ncol])
        fig = plt.figure(figsize=(4*n_sub, 4))
        i_plt = 1
        clean_x = pd.read_csv(DATA_PATH)
        _, pred_verbose = cal_TNPP(clean_x['prcp'].values, clean_x['tmean'].values, verbose=True)
        for i in range(n_sub):
            # Calculate the point density
            ax = fig.add_subplot(1, n_sub, i_plt)
            xy = np.vstack([fea_x[:, i], slope_arr[:, i]])
            z = gaussian_kde(xy)(xy)
            ax.scatter(fea_x[:, i], slope_arr[:, i], c=z)
            ax.set_xlabel(var_names[i])
            ax.set_ylabel(var_names[i] + ' slope')

            # calculate slope from equation model (dy/dx)
            if i < len(pred_verbose):
                ## dy_1 fea_x[:, i] to clean_x[var_names[i]]
                clean_x[var_names[i]] = clean_x[var_names[i]] - delta_x
                _, pred_verbose_1 = cal_TNPP(clean_x['prcp'].values, clean_x['tmean'].values,
                                           verbose=True)  # fea_x[:, 0] is prcp, fea_x[:, 1] is Tmean
                clean_x[var_names[i]] = clean_x[var_names[i]] + delta_x # restore fea_x[:, i]
                ## dy_2
                clean_x[var_names[i]] = clean_x[var_names[i]] + delta_x
                _, pred_verbose_2 = cal_TNPP(clean_x['prcp'].values, clean_x['tmean'].values,
                                           verbose=True)  # fea_x[:, 0] is prcp, fea_x[:, 1] is Tmean
                clean_x[var_names[i]] = clean_x[var_names[i]] - delta_x  # restore fea_x[:, i]
                plt_idx = (pred_verbose_1[i] != 0) * (pred_verbose_2[i] != 0) # avoid tip points
                ax.scatter(clean_x[var_names[i]][plt_idx], (pred_verbose_2[i][plt_idx] - pred_verbose_1[i][plt_idx]) / (2 * delta_x), s=0.5)

            i_plt += 1

        plt.tight_layout()
        if save_fig is not None:
            plt.savefig(os.path.join(SAVE_PATH, save_fig), dpi=400)
        else:
            plt.show()


def main(method_name='TNPP', nx=0, ny=0, niter=0, trans_pca=False, tune_model=True, noise_x='all'):
    """This is the main function for the simulation test
    Args:
        method_name (str, optional): A prefix to identify the simulation test. For the manuscript we use 'TNPP' to indicate the model.
        nx (int, optional): level of noise in X variable. The standard deviation of the noise is the noise level times the standard deviation of the corresponding variable.  Defaults to 0.
        ny (int, optional): level of noise in Y variable. The standard deviation of the noise is the noise level times the standard deviation of the corresponding variable. Defaults to 0.
        niter (int, optional): The nth time building a DNN model. This is an indicator for the variable selection test. Defaults to 0.
        trans_pca (bool, optional): If conduct the Principal component analysis (PCA) for each X variable. Defaults to False.
        tune_model (bool, optional): If tune model parameters using Hyperband package. Defaults to True.
        noise_x (str, optional): Which X variables to add noise. 'all' will add noise to all X variables. list of variable name will specify which variables to add noise. Defaults to 'all'.
    """
    test_data_path = DATA_PATH
    ## Create a folder to save all outputs ##
    if trans_pca:
        if noise_x == 'all':
            work_dir = os.path.join(SAVE_PATH, 'm{}_pca_nx{}_ny{}_iter{}'.format(method_name, nx, ny, niter))
        else:
            work_dir = os.path.join(SAVE_PATH,
                                    'm{}_pca_nx{}_ny{}_iter{}_{}'.format(method_name, nx, ny, niter, ''.join(noise_x)))
        make_save_path(work_dir)
    else:
        if noise_x == 'all':
            work_dir = os.path.join(SAVE_PATH, 'm{}_nx{}_ny{}_iter{}'.format(method_name, nx, ny, niter))
        else:
            work_dir = os.path.join(SAVE_PATH,
                                    'm{}_nx{}_ny{}_iter{}_{}'.format(method_name, nx, ny, niter, ''.join(noise_x)))
        make_save_path(work_dir)

    if os.path.exists(os.path.join(SAVE_PATH, 'simu_slope_var.png')): # if current iteration is done
        return

    ## Justify if this is the first run of the variable selection iteration. If yes, add noise to variables. If no, read data from the first iteration, so the inputs are the same for iterations. ##
    if niter == 0:
        #### Generate simulation data ####
        dataset, var_names = make_simudata_NPP(test_data_path, nx=nx, ny=ny, noise_x=noise_x)  # clean invalid X, calculate Y, add noise
        preview_data(dataset)
    else: # reuse iter0 inputs, so noises are the same
        test_data_path = os.path.join(os.path.dirname(SAVE_PATH), 'm{}_nx{}_ny{}_iter0'.format(method_name, nx, ny),
                                      'simu_all_fea_raw.csv')
        dataset = pd.read_csv(test_data_path) # load X with noise from iter0
        df_y = np.load(os.path.join(os.path.dirname(test_data_path), 'simu_y_raw.npy')) # load Y with noise from iter0
        dataset['y'] = df_y
        var_names = dataset.columns.values.tolist()

    ## Preprocessing: Shuffle X, Standerize (PCA), Split data ##
    if trans_pca:
        train_x, train_y, test_x, test_y, scaler, pca = preprocessing_simudata(dataset, pca=True, standerize=True, train_portion=0.8) # Shuffle X, Standerize (PCA), Split data
    else:
        train_x, train_y, test_x, test_y, scaler = preprocessing_simudata(dataset, pca=False, standerize=True, train_portion=0.8)
        pca = None
    
    ## Create DNN model. (1) tune model parameters or (2) load previously tuned model structure and build model ##
    if tune_model:
        tuner = tune_DNN_dataset(train_x, train_y, test_x, test_y, nx=nx, ny=ny) #nx and ny for file name only
        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_x, train_y, validation_split=0.5, verbose=0, epochs=30)
    else:
        ini_work_dir = os.path.join(SAVE_PATH, 'm{}_nx{}_ny{}_iter{}'.format(method_name, nx, ny, 0))
        model_arch_path = os.path.join(ini_work_dir, 'simu_model_architecture')
        model = make_DNN_simu_opt_json(model_arch_path)
        model, history = compile_fit_DNN_simu(model, train_x, train_y, return_history=True)

    ## Retrieve dataset ##
    print('vstack')
    all_fea = np.vstack((train_x, test_x)) # Preprocessed X data
    all_fea_raw = all_fea
    y_fea = np.vstack((train_y[:, np.newaxis], test_y[:, np.newaxis]))
    y_fea_raw = y_fea # Preprocessed X data
    if pca is not None:
        all_fea_raw = pca.inverse_transform(all_fea) # Restore original X data
    if scaler is not None:
        all_fea_raw = scaler.inverse_transform(all_fea_raw) # Restore original X data

    ## Export dataset ##
    np.save(os.path.join(SAVE_PATH, 'simu_all_fea_raw.npy'), all_fea_raw) # Export original X data
    df = pd.DataFrame(all_fea_raw, columns=var_names[:all_fea_raw.shape[1]])
    df.to_csv(os.path.join(SAVE_PATH, 'simu_all_fea_raw.csv'), index=False) # Export original X data with variable names
    np.save(os.path.join(SAVE_PATH, 'simu_model_pred.npy'), model.predict(all_fea)) # Export Y predictions for all data
    np.save(os.path.join(SAVE_PATH, 'simu_y_raw.npy'), y_fea_raw) # Export original Y data
    
    ## Export model architecture ##
    # print('Save model info')
    model_arch_path = os.path.join(SAVE_PATH, 'simu_model_architecture') # Export model architecture
    model_json = model.to_json()
    with open(model_arch_path, "w") as json_file:
        json.dump(model_json, json_file)
    model_weight_path = os.path.join(SAVE_PATH, 'simu_model_weights')
    model.save_weights(model_weight_path)
    
    ## Calculate variable contributions ##
    coef_arr = get_var_contrb_mp(all_fea, model_weight_path, ncpu=5) # calculate contribution for X (after preprocessing)
    np.save(os.path.join(SAVE_PATH, 'simu_coef_arr.npy'), coef_arr)
    contr_arr, bias = get_var_contr_raw(all_fea, coef_arr, scaler=scaler, pca=pca, return_bias=True) # calculate contribution for original X (before preprocessing)
    np.savez(os.path.join(SAVE_PATH, 'simu_contr_arr_bias'), contr=contr_arr, bias=bias)
    plt_contr_var(all_fea_raw, contr_arr, var_names, save_fig='simu_contr_var')
    slope_arr = get_var_slopes_raw(all_fea, coef_arr, scaler=scaler, pca=pca) # calculate slope for original X (before preprocessing)
    np.save(os.path.join(SAVE_PATH, 'simu_slope_arr.npy'), slope_arr)
    plt_slope_var(all_fea_raw, slope_arr, var_names, save_fig='simu_slope_var')


if __name__ == '__main__':
    ##  Run the noise-free simulation test  ##
    main(method_name='TNPP', nx=0, ny=0, niter=0, trans_pca=False, tune_model=True)
    
    ## ##  During the early test, I also implemented Principal component analysis (PCA) for preprocessing. 
    ## ##  PCA removes correlation between variables. ## ##
    ## ##  But I found there is minor affect of PCA processing, so I skiped the step in the study.
    ## ##  Please feel free to set the trans_pca to True to activate the preprocessing.  ## ##
    # main(method_name='TNPP', nx=0, ny=0, niter=0, trans_pca=True, tune_model=True)
    
    ##  Run the noise simulation test  ##
    # for ny in range(4):
    #     for nx in range(4):
    #         for niter in range(10):
    #             main(method_name='TNPP', nx=nx / 20.0, ny=ny / 20.0, niter=niter, trans_pca=False, tune_model=True) # add noise to Y and all X variables
    #             # main(method_name='TNPP', nx=nx / 20.0, ny=ny / 20.0, niter=niter, trans_pca=False, tune_model=True, noise_x=['prcp']) # add noise to Y and MAP variable

