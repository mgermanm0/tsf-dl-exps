import os
import pickle
from tsds import TimeSeriesDataSet
from notifier import telegram_bot_sendtext
from config_mgm import RAW_DATA_FOLDER, PREPO_DATA_FOLDER, PAST_HIST, CUSTOM_CONFIGS

ds_done = 0
for root, dirs, files in os.walk(RAW_DATA_FOLDER):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    
    for file in files:
        filepath = root + os.sep + file
        pkpath = PREPO_DATA_FOLDER + os.sep + path[-1]
        
        if not os.path.exists(pkpath):
            os.mkdir(pkpath)
            
        for mult in PAST_HIST:
            ds_path = pkpath + os.sep + f'past_hist_{mult}' + os.sep
            if not os.path.exists(ds_path):
                os.mkdir(ds_path) 
            ds = TimeSeriesDataSet(ds_path=filepath, name_ds=file, history=mult)
  
            dsdump = open(ds_path + file.split(".")[0] + ".pk", "wb")
            pickle.dump(ds, dsdump)
            dsdump.close()
            print(ds)
            if len(ds.series_train[0][0]) <= 0:
                print("empty")
            del ds
        ds_done+=1
      

telegram_bot_sendtext(f'Se han cargado {ds_done} datasets')
print(ds_done)