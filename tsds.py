import math
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config_mgm import SEASONALITY_MAP
from prepocess.yan import YanOutliersTransform
from slidingds import SlidingWindowDataset
from utils.tsf_data_loader import convert_tsf_to_dataframe


def delete_first_missing_vals(arr):
    to_del = arr[0]
    count = 0
    for i in range(1, to_del):
        if to_del == arr[i]:
            count += 1
        else: 
            return count, arr[count:]
        

class TimeSeriesDataSet():
    def __init__(self, ds_path, name_ds="ds", history=1.25) -> None:
        loaded_data, frequency, f_h, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(ds_path)
        self.freq = frequency
        self.name_ds = name_ds
        self.horizon = f_h
        self.equal_len = contain_equal_length
        self.missing_vals = contain_missing_values
        self.hist = math.ceil(history*SEASONALITY_MAP[frequency]) # seasonality * 1.25
        self.general_normalizer = MinMaxScaler() # general knowledge
        self.series_train = []
        self.series_test = []
        self.start_date = []
        self.normalizers = []
        
        for index, row in loaded_data.iterrows():
            series = row["series_value"].to_numpy()
            train, test = series[0:-f_h], series[len(series) - f_h - self.hist:]
            
            if(len(series.shape) < 2):
                train, test = train.reshape(-1, 1), test.reshape(-1, 1)
           
            # prepo aqui
            outliers = YanOutliersTransform(3)
            train_ol, test_ol = outliers.transform(train), outliers.transform(test)
            
            minmax = MinMaxScaler()
            minmax.fit(train_ol)


            self.normalizers.append(minmax)
            self.general_normalizer.partial_fit(train_ol)
            self.series_train.append(train_ol)
            self.series_test.append(test_ol)

            date = datetime(1990, 1, 1)
            if "start_timestamp" in row:
                date = row["start_timestamp"]
            self.start_date.append(date)

    def __str__(self) -> str:
        strout = f'TimeSeriesDataset(name_ds={self.name_ds}, num_series={len(self.series_train)}, freq={self.freq}, f_h={self.horizon}, hist_lags={self.hist})'
        return strout
    
class TimeSeriesCSVDataSet():
    def __init__(self, ds_path, name_ds="ds", history=96, forecast_horizon=1, test_size=100) -> None:
        loaded_data = pd.read_csv(ds_path)
        self.name_ds = name_ds
        self.horizon = forecast_horizon
        self.hist = history # seasonality * 1.25
        self.test_size = test_size
        self.general_normalizer = MinMaxScaler() # general knowledge
        self.series_train = []
        self.series_test = []
        self.start_date = []
        self.normalizers = []
        
        series = loaded_data["OT"].to_numpy()
        train, test = series[0:-self.test_size], series[len(series) - self.test_size - self.hist:]
        
        if(len(series.shape) < 2):
            train, test = train.reshape(-1, 1), test.reshape(-1, 1)
        
        # prepo aqui
        outliers = YanOutliersTransform(3)
        train_ol, test_ol = outliers.transform(train), outliers.transform(test)
        
        minmax = MinMaxScaler()
        minmax.fit(train_ol)


        self.normalizers.append(minmax)
        self.general_normalizer.partial_fit(train_ol)
        self.series_train.append(train_ol)
        self.series_test.append(test_ol)

        date = datetime(1990, 1, 1)
        if "date" in loaded_data:
            date = loaded_data["date"].iloc[0]
        self.start_date.append(date)

    def __str__(self) -> str:
        strout = f'TimeSeriesCSVDataset(name_ds={self.name_ds}, num_series={len(self.series_train)}, f_h={self.horizon}, hist_lags={self.hist})'
        return strout