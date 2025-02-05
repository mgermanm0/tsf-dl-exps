import argparse
import itertools
import os
import pickle
import sys
import time

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Get the parent directory (tfm) of the current script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
tfm_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python search path
sys.path.append(tfm_dir)

import config_mgm
from models.tcn import TemporalConvNet
from rescollector import read_results_file
from slidingds import SlidingWindowDataset

print(torch.cuda.is_available())


torch.manual_seed(config_mgm.SEED)
torch.cuda.manual_seed(config_mgm.SEED)
torch.cuda.manual_seed_all(config_mgm.SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[TFM-MGM] Train and test TCN models on specified dataset. General approach.')
    parser.add_argument('-ds','--dataset', metavar='DS', type=str,
                        help='Dataset used for experimentarion. Mandatory.')

    args = parser.parse_args()
    ds_key = args.dataset.lower()
    if ds_key is None:
        raise TypeError("Dataset value is None. Did you specify it using -ds?")
    if ds_key not in config_mgm.PREPO_DS_PATHS:
        raise ValueError(f'{ds_key} is not a valid dataset. Valid values: {list(config_mgm.PREPO_DS_PATHS.keys())}')
    hyperparams = config_mgm.MODELS_HYPERPARAMS["tcn"]
    columns = ["DATASET", "PAST_HIST", "MODEL", "LAYERS", "UNITS", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "RMSE", "SMAPE", "WAPE", "TRAIN_TIME_NS", "INFERENCE_TIME_NS"]
    for root, dirs, files in os.walk(config_mgm.PREPO_DS_PATHS[ds_key]):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            fileds = open(root + os.sep + file, 'rb')
            past_hist = path[-1].split("_")[-1]
            #plots_folder = config_mgm.PLOTS_FOLDER + os.sep + os.sep.join(path[1:])
            res_folder = config_mgm.RESULTS_FOLDER + os.sep + os.sep.join(path[1:])
            #os.makedirs(plots_folder, exist_ok=True)
            os.makedirs(res_folder, exist_ok=True)
            ds = pickle.load(fileds)
            fileds.close()
            trainds = SlidingWindowDataset(ds.series_train, ds.hist, ds.horizon, normalizers=[ds.general_normalizer])
            testds = SlidingWindowDataset(ds.series_test, ds.hist, ds.horizon, normalizers=[ds.general_normalizer])
            
            ef_bs_multiplier = config_mgm.GPUS_TO_USE if config_mgm.GPUS_TO_USE > 0 else 1

            resdf = read_results_file(res_folder + os.sep + "results_general_model_tcn.csv", columns)
            for epochs, tcn_layers, tcn_size, dropout, bs, lr in itertools.product(hyperparams["epochs"], hyperparams["layers"], hyperparams["hidden_size"],hyperparams["dropout"], hyperparams["bs"], hyperparams["lr"]):
                modelname = f'tcn_layers_{tcn_layers}_size_{tcn_size}_epochs_{epochs}_dropout_{dropout}_bs_{bs}_lr_{lr}'
                print(f'\t\t{modelname}')
                
                traindl = DataLoader(trainds, bs * ef_bs_multiplier, True)
                testdl = DataLoader(testds, bs) #len(testds)
                
                tcn = TemporalConvNet(input_feat=1, tcn_hidden_size=tcn_size, tcn_n_layers=tcn_layers, tcn_dropout=dropout, out_size=ds.horizon, lr=lr)

                logger = TensorBoardLogger(save_dir=f'./lightning_logs/general/{file}_logs/', name=f'{file}_{modelname}')
                trainer = None
                if config_mgm.GPUS_TO_USE <= 0:
                    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=epochs, logger=False)
                elif config_mgm.GPUS_TO_USE == 1:
                    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=epochs, accelerator="gpu", logger=logger)
                else:
                    trainer = pl.Trainer(log_every_n_steps=1, max_epochs=epochs, accelerator="gpu", strategy="ddp", devices=config_mgm.GPUS_TO_USE, logger=False)
                
                train_start = time.perf_counter_ns()
                trainer.fit(tcn, train_dataloaders=traindl)
                train_stop = time.perf_counter_ns()
                train_time = train_stop - train_start
                
                inference_start = time.perf_counter_ns()
                res = trainer.test(tcn, testdl, verbose=True)[0]
                inference_stop = time.perf_counter_ns()
                inference_time = inference_stop - inference_start
                
                loss, mse, rmse, smape, wape = res['test_loss'], res['test_mse'], res['test_rmse'], res['test_smape'], res['test_wape']
                
                res_dict = {
                    "DATASET": [file],
                    "PAST_HIST": [past_hist],
                    "MODEL": [f'TCN_dropout_{dropout}'],
                    "LAYERS": [tcn_layers],
                    "UNITS": [tcn_size],
                    "EPOCHS": [epochs],
                    "BATCH_SIZE": [bs],
                    "LEARNING_RATE": [lr],
                    "RMSE": [rmse],
                    "SMAPE": [smape],
                    "WAPE": [wape],
                    "TRAIN_TIME_NS": [train_time],
                    "INFERENCE_TIME_NS": [inference_time]
                }
                resdf = pd.concat([resdf, pd.Series(res_dict).to_frame().T], ignore_index=True)
            resdf.to_csv(res_folder + os.sep + "results_general_model_tcn.csv", sep=";")