#!/panfs/ccds02/nobackup/people/qzhou2/miniconda3/envs/dnn/bin/python
import sys
import platform

import numpy as np
import pandas as pd
import os
from datetime import datetime
from glob import glob
from sklearn.decomposition import PCA
from joblib import dump, load

from sklearn import preprocessing
from scipy.stats import linregress

import pathlib

# from math import sin, cos, tan
# import sympy
# from sympy import *

# import time
# import tqdm

# import json
import pickle as pkl

# import feyn # does not return sin cos function
from pysr import PySRRegressor
from pysr import jl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import gaussian_kde

if platform.system() == "Linux":
    DATA_DIR = r"/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/TNPP" # r'/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/Data'
    SAVE_PATH = r"/home/qzhou2/DNN" # r'/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN'
elif platform.system() == "Windows":
    # DATA_DIR = r'C:\Users\qzhou2\C_workdir\DNN_REMODEL\DNN\Penman–Monteith\output'
    SAVE_PATH = r'C:\Users\qzhou2\C_workdir\DNN_REMODEL\DNN'

N_FEATURE = 15
def make_save_path(new_save_path):
    global SAVE_PATH
    SAVE_PATH = new_save_path
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)


def preview_data(dataset):
    print('dataset.describe(): ', dataset.describe())
    print('Correlation: ', dataset.corr(method='pearson'))

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


# def true_PET(df):
#     return (df.delta*(df.Rn-df.G)+1.205*1004*(df.es-df.ea)/df.ra)/((df.delta+0.066)*2.45*10**6*1000)*3600*1000


def preprocessing_data(data_path:str, pca=True, standerize=True,train_portion=0.8, min_train=3000):
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
    nx = float(os.path.basename(data_path).split('_')[1][2:])
    ny = float(os.path.basename(data_path).split('_')[2][2:])
    df_slope = np.load(os.path.join(data_path, 'simu_slope_arr.npy'))
    prcp_slope_idx = df_slope[:, 0] > 0.2
    x_df = pd.read_csv(os.path.join(data_path, f'simu_all_fea_raw.csv'))
    x_qa_idx = np.asarray([True]*len(x_df))
    for col in x_df.columns:
        if (nx != 0) and (col in ["prcp", "tmean"]):
            x_qa_idx = x_qa_idx * ((x_df[col]/(nx*np.std(x_df[col]))) > 20)
    fit_idx = prcp_slope_idx * x_qa_idx
    print(f"fit_idx: {np.sum(fit_idx)}")
    x_df = x_df[fit_idx]
    var_names = x_df.columns
    x_stack = x_df.to_numpy()
    print('x_stack.shape: ', x_stack.shape) # x_stack.shape:  (260388, 15)
    y_var = np.load(os.path.join(data_path, f'simu_y_raw.npy'))[fit_idx]
    suf_idx = np.arange(x_stack.shape[0])
    np.random.shuffle(suf_idx)
    x_stack = x_stack[suf_idx, :]
    y_var = y_var[suf_idx]
    n_training = max(int(x_stack.shape[0] * train_portion), min_train)
    print('n_training: ', n_training)
    if standerize:
        # scaler = preprocessing.StandardScaler()
        scaler = load(os.path.join(data_path, "scaler.bin"))
        x_stack = scaler.fit_transform(x_stack)

    else:
        scaler = None

    if pca:
        pca_model = PCA()  # n_components=2
        x_stack = pca_model.fit_transform(x_stack)
        # print('x_pca.shape=', x_pca.shape)

    train_x = x_stack[:n_training, :].copy()
    train_y = y_var[:n_training].copy()

    test_x = x_stack[n_training:, :].copy()
    test_y = y_var[n_training:].copy()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    if pca:
        return var_names, train_x, train_y, test_x, test_y, scaler, pca_model
    return var_names, train_x, train_y, test_x, test_y, scaler


def fit_PySRRegressor(data_path: str, sub_folder: str):
    # s_band = "B8A"
    var_names, train_x, train_y, _, _, _ = preprocessing_data(data_path=data_path, pca=False, standerize=False, train_portion=0.1, min_train=3000) #'B02'
    df = pd.DataFrame(data=train_x, columns=var_names)
    if not os.path.exists(os.path.join(data_path, sub_folder)):
        os.makedirs(os.path.join(data_path, sub_folder))
    # print(train_x.shape, train_y.shape)
    df['response'] = train_y
    # print(df.head)

    # jl.seval("""
    # import Pkg
    # Pkg.add("NPZ")
    # Pkg.add("DataFrames")
    # """)
    jl.seval("using NPZ")
    jl.seval("using DataFrames")
    # jl.seval("""der_val = transpose(npzread("/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/model/S2L_B02_iter0/slope_arr_comb.npy"))""")
    # jl.seval("""der_val = transpose(npzread("C:/Users/qzhou2/C_workdir/Goden_tile/DNN/slope_arr_comb.npy"))""")
    
    objective = """
    function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        (prediction, completion) = eval_tree_array(tree, dataset.X, options)
        if !completion
            return L(Inf)
        end
        diffs = prediction .- dataset.y
        rmse = sqrt(sum(diffs .^ 2) / length(diffs))

        # using NPZ
        # using DataFrames
        der_val = transpose(npzread("C:/Users/qzhou2/C_workdir/Goden_tile/DNN/slope_arr_comb.npy"))
        # der_val = transpose(npzread("/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/model/S2L_B02_iter0/slope_arr_comb.npy"))
        der_val = der_val[:, 1:size(dataset.X)[2]]

        step=0.001
        loss_der = Vector{Float64}()
        for n in 1:size(dataset.X)[1]
            dataset.X[n, :] .-= step
            (pred_pre, completion) = eval_tree_array(tree, dataset.X, options)
            if !completion
                return L(Inf)
            end
            dataset.X[n, :] .+= 2 * step
            (pred_pos, completion) = eval_tree_array(tree, dataset.X, options)
            if !completion
                return L(Inf)
            end
            dataset.X[n, :] .-= step

            if (sum(abs.(pred_pre.-pred_pos))!=0) || (sum(abs.(pred_pre.-prediction))!=0) || (sum(abs.(prediction.-pred_pos))!=0)
                der = (pred_pos .- pred_pre)./(2*step)
                diffs = der .- der_val[n, :]
                rmse_der = sqrt(sum(diffs .^ 2) / length(diffs))
                # println(size(der), size(diffs), size(rmse_der), rmse_der)
                push!(loss_der, rmse_der)
            end
        end
        # return rmse
        return rmse + sum(loss_der) #/length(loss_der)
    end
    """
    dt_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # https://github.com/MilesCranmer/PySR/blob/master/pysr/sr.py
    model = PySRRegressor(
        niterations=10000,  # < Increase me for better results
        maxsize=30, # Max complexity of an equation.  Default is `20`.
        binary_operators=["+", "-", "*", "/", "^"], #
        unary_operators=[
            "exp",
            # "log",
            # "sqrt",
        ],
        # extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # loss_function=objective,
        # ^ Custom loss function (julia syntax)
        # batching=True,
        equation_file=os.path.join(data_path, sub_folder, f"equation_{dt_str}.csv"), #  Where to save the files (.csv extension). Default is `None`.
        tempdir=data_path # directory for the temporary files.
    )

    model.fit(df.drop(['response'], axis=1), df['response'])
    # print(model)


# def eq_julia2python(eq: str):
#     # opp_dict = {"^": "**",
#     #             "sin": "np.sin",
#     #             "cos": "np.cos",
#     #             "exp": "np.exp",
#     #             "tan": "np.tan"}
#     opp_dict = {"^": "**"}
#     for k in opp_dict.keys():
#         eq = eq.replace(k, opp_dict[k])
#     return eq


def eval_PySRRegressor(iter_folder: str, sub_folder: str):
    model_dir = os.path.join(DATA_DIR, iter_folder)
    eq_files = glob(os.path.join(model_dir, sub_folder, "equation_*.pkl"))
    if len(eq_files)==0:
        print(f"eval_PySRRegressor: No equation found in {model_dir} {sub_folder}")
        return

    df = pd.read_csv(os.path.join(model_dir, 'simu_all_fea_raw.csv'))
    df_slope = np.load(os.path.join(model_dir, 'simu_slope_arr.npy'))
    train_y = np.load(os.path.join(model_dir, 'simu_y_raw.npy')).squeeze()
    prcp_idx = df_slope[:, 0] > 0.2
    x_qa_idx = np.asarray([True]*len(df))
    nx = float(os.path.basename(model_dir).split('_')[1][2:])
    for col in df.columns:
        if (nx != 0) and (col in ["prcp", "tmean"]):
            x_qa_idx = x_qa_idx * ((df[col]/(nx*np.std(df[col]))) > 20)
    fit_idx = prcp_idx * x_qa_idx
    print(f"nx: {nx}, fit_idx: {np.sum(fit_idx)}")
    # df = df[fit_idx]
    # df_slope = df_slope[fit_idx, :]
    # train_y = train_y[fit_idx]
    model_eval = []
    for m in eq_files:
        # print(m)
        # eq_path = os.path.join(data_dir, m)
        model = load_model(m)
        # print(model.equations_)
        for i in range(len(model.equations_)):
            mad = np.mean(np.abs(model.predict(df, i) - train_y))
            rmse_std = np.sqrt(np.mean((model.predict(df, i) - train_y)**2)) / np.std(train_y)
            loss_der = eval_eq_str_slope(model, i, df, df_slope, fit_idx)
            if loss_der is None:
                model_eval.append({"model_path": m, "model_index": i, "Equation": model.equations_.sympy_format[i], "Loss": model.equations_.loss[i], "Loss_der": None, "MAD": mad, "RMSE_std": rmse_std, "newScore": None})
            else:    
                model_eval.append({"model_path": m, "model_index": i, "Equation": model.equations_.sympy_format[i], "Score": model.equations_.score[i], "Loss": model.equations_.loss[i], "Loss_der": loss_der, "MAD": mad, "RMSE_std": rmse_std, "newScore": (loss_der + rmse_std)})
            # print(eval_eq_str_slope(eq_path, i, df, train_y, df_slope))
    df_model_eval = pd.DataFrame(model_eval)
    # print("Summary df: \n", df.head(), "\n")
    df_model_eval.to_csv(os.path.join(model_dir, sub_folder, f"model_summary_{iter_folder}.csv"))

    max_score = df_model_eval["Score"].idxmax()
    # print(f"{iter_folder} Original best model: ", load_model(df_model_eval.loc[max_score]["model_path"]).equations_.sympy_format[df_model_eval.loc[max_score]["model_index"]])
    # print("New best model index: \n", model.equations_.head(), "\n")
    min_new = df_model_eval["newScore"].idxmin()
    # print("New best model: \n", df.loc[min_der], "\n")
    # print(f"{iter_folder} New best model: \n", load_model(df_model_eval.loc[min_new]["model_path"]).equations_.sympy_format[df_model_eval.loc[min_new]["model_index"]], "\n")
    with open(os.path.join(DATA_DIR, "sym_model.txt"), "a") as file:
        org_model = load_model(df_model_eval.loc[max_score]["model_path"]).equations_.sympy_format[df_model_eval.loc[max_score]["model_index"]]
        new_model = load_model(df_model_eval.loc[min_new]["model_path"]).equations_.sympy_format[df_model_eval.loc[min_new]["model_index"]]
        file.writelines(f"Original, {iter_folder}, {org_model}\n")
        file.writelines(f"New, {iter_folder}, {new_model}\n")
    ## Validation ##
    org_pred = load_model(df_model_eval.loc[max_score]["model_path"]).predict(df, df_model_eval.loc[max_score]["model_index"])
    new_pred = load_model(df_model_eval.loc[min_new]["model_path"]).predict(df, df_model_eval.loc[min_new]["model_index"])
    org_slope_name = os.path.join(model_dir, sub_folder, f"org_symbolic_slope.csv")
    eval_eq_str_slope(load_model(df_model_eval.loc[max_score]["model_path"]), df_model_eval.loc[max_score]["model_index"], df, df_slope, fit_idx, slope_out=org_slope_name)
    new_slope_name = os.path.join(model_dir, sub_folder, f"new_symbolic_slope.csv")
    eval_eq_str_slope(load_model(df_model_eval.loc[min_new]["model_path"]), df_model_eval.loc[min_new]["model_index"], df, df_slope, fit_idx, slope_out=new_slope_name)
    sym_org_rmse = np.sqrt(np.mean(np.square(train_y - org_pred)))
    sym_new_rmse = np.sqrt(np.mean(np.square(train_y - new_pred)))
    dnn_pred = np.load(os.path.join(DATA_DIR, iter_folder, "simu_model_pred.npy")).flatten()# [fit_idx]
    dnn_rmse = np.sqrt(np.mean(np.square(train_y - dnn_pred))) #.values.flatten()
    print(f"{iter_folder} Sym org RMSE:", sym_org_rmse, " Sym new RMSE:", sym_new_rmse, " DNN RMSE:", dnn_rmse)
    pd.DataFrame({"DNN_true": train_y, "DNN_pred": dnn_pred, "Sym_org": org_pred, "Sym_new": new_pred, "Sym_idx": fit_idx}).to_csv(os.path.join(DATA_DIR, iter_folder, f"{iter_folder}_pred_compare.csv"))
    



def load_model(eq_path: str):
    # posix_backup = pathlib.PosixPath
    # try:
    #     pathlib.PosixPath = pathlib.WindowsPath
    #     # https://github.com/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb
    #     model = PySRRegressor.from_file(eq_path)
    #     # print(model.__dict__)
    #     # print(model.equations_.head())
    #     # print("type of model.equations_: ", type(model.equations_))
    # finally:
    #     pathlib.PosixPath = posix_backup
    return PySRRegressor.from_file(eq_path) # model


def eval_eq_str_slope(model, eq_idx:int, df_x, df_slope, valid_idx, step=0.0001, slope_out=None)->float:
    # model = load_model(eq_path)
    # print(model.equations_.sympy_format[eq_idx])
    eq_slope = df_x.copy()
    loss_der = []
    for n in range(len(df_x.columns)):
        df_x.iloc[:, n] -= step
        pred_pre = model.predict(df_x, eq_idx)
        df_x.iloc[:, n] += 2 * step
        pred_pos = model.predict(df_x, eq_idx)
        df_x.iloc[:, n] -= step
        df_y = model.predict(df_x, eq_idx)
        # print(df_x.columns[n], sum(abs(pred_pre-df_y)), sum(abs(df_y-pred_pos)))
        if (sum(abs(pred_pre-df_y))!=0) | (sum(abs(df_y-pred_pos))!=0): #(sum(abs(pred_pre-pred_pos))!=0) | 
            der = (pred_pos - pred_pre)/(2*step)
            eq_slope.iloc[:, n] = der
            diffs = der[valid_idx] - df_slope[valid_idx, n]
            rmse_der = np.sqrt(sum(diffs ** 2) / len(diffs)) / np.std(df_slope[valid_idx, n]) # np.std(df_x.iloc[:, n])
            # print(df_x.columns[n], rmse_der)
            # println(size(der), size(diffs), size(rmse_der), rmse_der)
            loss_der.append(rmse_der)
        else:
            eq_slope.iloc[:, n] *= 0
    if slope_out:
        eq_slope.to_csv(slope_out)
    # return rmse
    # print(model.equations_["equation"][eq_idx], loss_der)
    if len(loss_der) == 0:
        return None
    return sum(loss_der) /len(loss_der)


def extract_final_eq(sub_folder: str):
    df = pd.DataFrame(columns=["Scenario", "Symbolic", "DNN+Symbolic"])
    for ny in range(4):
        for nx in range(4):
            iter_folder = f"mTNPP_nx{nx / 20.0}_ny{ny / 20.0}_iter0"
            model_dir = os.path.join(DATA_DIR, iter_folder)
            model_dir = os.path.join(model_dir, sub_folder, f"model_summary_{iter_folder}.csv")
            if not os.path.exists(model_dir):
                print(f"extract_final_eq: No model found in {model_dir}")
                continue
            df_model = pd.read_csv(model_dir)
            max_score = df_model["Score"].idxmax()
            min_new = df_model["newScore"].idxmin()
            org_model = df_model.loc[max_score]["Equation"]
            new_model = df_model.loc[min_new]["Equation"]
            rec = pd.DataFrame([{"Scenario": f"$noise_x$={nx/20.0}, $noise_y$={ny/20.0}", "Symbolic": org_model, "DNN+Symbolic": new_model}])
            df = pd.concat([df, rec], ignore_index=True)
    df.to_csv(os.path.join(DATA_DIR, "final_eq.csv"), index=False)


def plt_slope_comparison(sub_folder):
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for ny in range(4):
        for nx in range(4):
            iter_folder = f"mTNPP_nx{nx / 20.0}_ny{ny / 20.0}_iter0"
            model_dir = os.path.join(DATA_DIR, iter_folder)
            df = pd.read_csv(os.path.join(model_dir, 'simu_all_fea_raw.csv'))
            train_y = np.load(os.path.join(model_dir, 'simu_y_raw.npy')).squeeze()
            df_slope_dnn = np.load(os.path.join(model_dir, 'simu_slope_arr.npy'))

            prcp_idx = df_slope_dnn[:, 0] > 0.2
            df = df[prcp_idx]
            train_y = train_y[prcp_idx]
            df_slope_dnn = df_slope_dnn[prcp_idx, :]
            # if (nx == 0) and (ny == 0):
            y_range = (min(df_slope_dnn[:, 0]), max(df_slope_dnn[:, 0]))
            # calculate slope from equation model (dy/dx)
            delta_x = 0.1
            df["prcp"] = df["prcp"] - delta_x
            _, pred_verbose_1 = cal_TNPP(df['prcp'].values, df['tmean'].values,
                                        verbose=True)  # fea_x[:, 0] is prcp, fea_x[:, 1] is Tmean
            df["prcp"] = df["prcp"] + delta_x # restore fea_x[:, i]
            ## dy_2
            df["prcp"] = df["prcp"] + delta_x
            _, pred_verbose_2 = cal_TNPP(df['prcp'].values, df['tmean'].values,
                                        verbose=True)  # fea_x[:, 0] is prcp, fea_x[:, 1] is Tmean
            df["prcp"] = df["prcp"] - delta_x  # restore fea_x[:, i]
            plt_idx = (pred_verbose_1[0] != 0) * (pred_verbose_2[0] != 0) # avoid tip points
            x_true, y_true = df["prcp"][plt_idx], (pred_verbose_2[0][plt_idx] - pred_verbose_1[0][plt_idx]) / (2 * delta_x)

            df_slope_org = pd.read_csv(os.path.join(model_dir, sub_folder, 'org_symbolic_slope.csv'))
            df_slope_new = pd.read_csv(os.path.join(model_dir, sub_folder, 'new_symbolic_slope.csv'))
            xy = np.vstack([df["prcp"], df_slope_dnn[:, 0]])
            z = gaussian_kde(xy)(xy)
            axs[ny, nx].scatter(df["prcp"], df_slope_dnn[:, 0], c=z, cmap='cividis')
            axs[ny, nx].scatter(x_true, y_true, c="red", label="True", s=0.5)
            axs[ny, nx].scatter(df["prcp"], df_slope_org["prcp"][prcp_idx], c="darkorange", label="Symbolic", s=0.5)
            axs[ny, nx].scatter(df["prcp"], df_slope_new["prcp"][prcp_idx], c="green", label="DNN+Symbolic", s=0.5)
            _, _, r_sym, _, _ = linregress(y_true, df_slope_org["prcp"][prcp_idx][plt_idx]) #slope, intercept, r_value, p_value, std_err
            _, _, r_new, _, _ = linregress(y_true, df_slope_new["prcp"][prcp_idx][plt_idx]) #slope, intercept, r_value, p_value, std_err
            rmse_sym = np.sqrt(np.mean(np.square(y_true - df_slope_org["prcp"][prcp_idx][plt_idx])))
            rmse_new = np.sqrt(np.mean(np.square(y_true - df_slope_new["prcp"][prcp_idx][plt_idx])))
            # axs[nx, ny].annotate(f"$R^2$ = {np.round(r_sym, 2)}", xy=(0.53, 0.9), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[ny, nx].annotate(f"$R^2$ = {r_sym: .2f}", xy=(0.45, 0.9), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[ny, nx].annotate(f"$RMSE$ = {rmse_sym: .1e}", xy=(0.45, 0.8), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[ny, nx].annotate(f"$R^2$ = {r_new: .2f}", xy=(0.45, 0.7), xycoords='axes fraction',color='green',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[ny, nx].annotate(f"$RMSE$ = {rmse_new: .1e}", xy=(0.45, 0.6), xycoords='axes fraction',color='green',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

            axs[ny, nx].set_ylim(y_range)
            if ny == 3:
                axs[ny, nx].set_xlabel("MAP")
            if nx == 0:
                axs[ny, nx].set_ylabel("Model slope")
            axs[ny, nx].set_title(f"$noise_x$={nx/20.0}, $noise_y$={ny/20.0}")
            if (nx == 0) and (ny == 0):
                axs[ny, nx].legend(markerscale=10, loc='lower left')
            
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plt_slope_comparison.png"), dpi=400)


def plt_overall_comparison(sub_folder):
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for ny in range(4):
        for nx in range(4):
            iter_folder = f"mTNPP_nx{nx / 20.0}_ny{ny / 20.0}_iter0"
            model_dir = os.path.join(DATA_DIR, iter_folder)
            df = pd.read_csv(os.path.join(model_dir, f'{iter_folder}_pred_compare.csv'))
            df_slope_dnn = np.load(os.path.join(model_dir, 'simu_slope_arr.npy'))
            prcp_idx = df_slope_dnn[:, 0] > 0.2
            df = df[prcp_idx]
            # df = df[df["Sym_idx"]]
            y_range = (min(df["DNN_true"]), max(df["DNN_true"]))
            xy = np.vstack([df["DNN_true"], df["DNN_pred"]])
            z = gaussian_kde(xy)(xy)
            axs[nx, ny].scatter(df["DNN_true"], df["DNN_pred"], c=z, cmap='cividis')
            axs[nx, ny].scatter(df["DNN_true"], df["Sym_org"], c="darkorange", label="Symbolic", s=0.5)
            axs[nx, ny].scatter(df["DNN_true"], df["Sym_new"], c="green", label="DNN+Symbolic", s=0.5)
            axs[nx, ny].scatter(df["DNN_true"], df["DNN_true"], c="red", label="True", s=0.5)
            _, _, r_sym, _, _ = linregress(df["DNN_true"], df["Sym_org"]) #slope, intercept, r_value, p_value, std_err
            _, _, r_new, _, _ = linregress(df["DNN_true"], df["Sym_new"]) #slope, intercept, r_value, p_value, std_err
            _, _, r_dnn, _, _ = linregress(df["DNN_true"], df["DNN_pred"]) #slope, intercept, r_value, p_value, std_err
            rmse_sym = np.sqrt(np.mean(np.square(df["DNN_true"] - df["Sym_org"])))
            rmse_new = np.sqrt(np.mean(np.square(df["DNN_true"] - df["Sym_new"])))
            rmse_dnn = np.sqrt(np.mean(np.square(df["DNN_true"] - df["DNN_pred"])))
            # axs[nx, ny].annotate(f"$R^2$ = {np.round(r_sym, 2)}", xy=(0.53, 0.9), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$R^2$ = {r_sym**2: .2f}", xy=(0.05, 0.9), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$RMSE$ = {rmse_sym: .2f}", xy=(0.05, 0.8), xycoords='axes fraction',color='darkorange',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$R^2$ = {r_new**2: .2f}", xy=(0.05, 0.7), xycoords='axes fraction',color='green',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$RMSE$ = {rmse_new: .2f}", xy=(0.05, 0.6), xycoords='axes fraction',color='green',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$R^2$ = {r_dnn**2: .2f}", xy=(0.05, 0.5), xycoords='axes fraction',color='blue',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            axs[nx, ny].annotate(f"$RMSE$ = {rmse_dnn: .2f}", xy=(0.05, 0.4), xycoords='axes fraction',color='blue',bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

            axs[nx, ny].set_ylim(y_range)
            if nx == 3:
                axs[nx, ny].set_xlabel("TNPP true")
            if ny == 0:
                axs[nx, ny].set_ylabel("Model prediction")
            axs[nx, ny].set_title(f"$noise_x$={nx/20.0}, $noise_y$={ny/20.0}")
            if (nx == 0) and (ny == 0):
                axs[nx, ny].legend(markerscale=10, loc='lower right')
            
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "plt_overall_comparison_TNPP.png"), dpi=400)


if __name__ == '__main__':
    ################ Test: Symbolic Regression  ######################
    # feyn_test() # Not working
    # pysr_test() # Found!
    # gplearn_test() # found equation but redundant

    ################  Symbolic Regression PySRRegressor  ######################
    parlist_idx = int(sys.argv[1])
    parlist = []
    for ny in range(4):
        for nx in range(4):
            parlist.append((nx / 20.0, ny / 20.0))
    x, y = parlist[parlist_idx]
    for i in range(10):
        # fit_PySRRegressor(data_path=f"/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/Penman–Monteith/output/mPET_nx0.0_ny0.0_iter0")
        fit_PySRRegressor(data_path=f"/panfs/ccds02/nobackup/projects/hls/HLS/qzhou2/DNN/TNPP/mTNPP_nx{x}_ny{y}_iter0", sub_folder="highSNR")

    ###############  PySRRegressor new scores ######################
    sub_folder = "highSNR"
    for ny in range(4):
        for nx in range(4):
            iter_folder = f"mTNPP_nx{nx / 20.0}_ny{ny / 20.0}_iter0"
            eval_PySRRegressor(iter_folder, sub_folder)
    # plt_slope_comparison(sub_folder)
    # plt_overall_comparison(sub_folder)
    extract_final_eq(sub_folder)
