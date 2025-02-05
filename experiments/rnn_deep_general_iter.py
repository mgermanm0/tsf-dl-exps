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

current_dir = os.path.dirname(os.path.abspath(__file__))
tfm_dir = os.path.dirname(current_dir)

sys.path.append(tfm_dir)

import config_mgm  # noqa: E402
from models.rnn import MIMORNN  # noqa: E402
from rescollector import read_results_file  # noqa: E402
from slidingds import SlidingWindowDataset  # noqa: E402



torch.manual_seed(config_mgm.SEED)
torch.cuda.manual_seed(config_mgm.SEED)
torch.cuda.manual_seed_all(config_mgm.SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[TFM-MGM] Train and test RNN-Based models on specified dataset. General approach.')
    parser.add_argument('-ds','--dataset', metavar='DS', type=str,
                        help='Dataset used for experimentarion. Mandatory.')

    args = parser.parse_args()
    ds_key = args.dataset.lower()
    if ds_key is None:
        raise TypeError("Dataset value is None. Did you specify it using -ds?")
    if ds_key not in config_mgm.PREPO_DS_PATHS:
        raise ValueError(f'{ds_key} is not a valid dataset. Valid values: {list(config_mgm.PREPO_DS_PATHS.keys())}')
    print(torch.cuda.is_available())
    print("=====> GPUS TO USE " + str(config_mgm.GPUS_TO_USE))
    hyperparams = config_mgm.MODELS_HYPERPARAMS["rnn"]
    product_hyperparams = itertools.product(hyperparams["flavor"], hyperparams["epochs"], hyperparams["layers"], hyperparams["hidden_units"], hyperparams["bs"], hyperparams["lr"])
    columns = ["DATASET", "PAST_HIST", "MODEL", "LAYERS", "UNITS", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "RMSE", "SMAPE", "WAPE", "TRAIN_TIME_NS", "INFERENCE_TIME_NS"]
    print("Starting experiment for dataset: " + config_mgm.PREPO_DS_PATHS[ds_key])
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
            
            resdf = read_results_file(res_folder + os.sep + "results_general_model_rnn.csv", columns)
            for flavor, epochs, rnn_layers, hidden_units, bs, lr in product_hyperparams:
                modelname = f'rnn_{flavor}_layers_{rnn_layers}_units_{hidden_units}_epochs_{epochs}_bs_{bs}_lr_{lr}'
                print(f'\t\t{modelname}')
                
                traindl = DataLoader(trainds, bs * ef_bs_multiplier, shuffle=True, num_workers=2)
                testdl = DataLoader(testds, bs) # len(testds)
                
                rnn = MIMORNN(num_rnn_layers=rnn_layers, hidden_dim=hidden_units, input_features=1, 
                            input_size=ds.hist, output_size=ds.horizon, flavor=flavor, return_sequences=True, lr=lr)
                logger = TensorBoardLogger(save_dir=f'./lightning_logs/general/{file}_logs/', name=f'{file}_{modelname}')
                trainer = None
                if config_mgm.GPUS_TO_USE <= 0:
                    trainer = pl.Trainer(log_every_n_steps=50, max_epochs=epochs, logger=False)
                elif config_mgm.GPUS_TO_USE == 1:
                    trainer = pl.Trainer(log_every_n_steps=50, max_epochs=epochs, accelerator="gpu", logger=logger)
                else:
                    trainer = pl.Trainer(log_every_n_steps=50, max_epochs=epochs, accelerator="gpu", strategy="ddp", devices=config_mgm.GPUS_TO_USE, logger=False)
                
                train_start = time.perf_counter_ns()
                trainer.fit(rnn, train_dataloaders=traindl)
                train_stop = time.perf_counter_ns()
                train_time = train_stop - train_start
                
                inference_start = time.perf_counter_ns()
                res = trainer.test(rnn, dataloaders=testdl, verbose=True, )[0]
                inference_stop = time.perf_counter_ns()
                inference_time = inference_stop - inference_start
                
                loss, mse, rmse, smape, wape = res['test_loss'], res['test_mse'], res['test_rmse'], res['test_smape'], res['test_wape']
                
                res_dict = {
                    "DATASET": [file],
                    "PAST_HIST": [past_hist],
                    "MODEL": [flavor],
                    "LAYERS": [rnn_layers],
                    "UNITS": [hidden_units],
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
            resdf.to_csv(res_folder + os.sep + "results_general_model_rnn.csv", sep=";")
    print("Experiment finished.")