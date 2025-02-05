from os.path import relpath
# Semilla
SEED = 57764

# Multiplicador para crear el horizonte de prediccion
PAST_HIST = [1.25]
CUSTOM_CONFIGS = {
   'etth1': [11521, 96, 512], #test size, forecast horizon, input size
   'etth2': [11521, 96, 512],
   'ettm1': [2881, 96, 512],
   'ettm2': [11521, 96, 512],
   'ili': [170, 24, 96],
   'weather': [11521, 96, 512],
} 

# Rutas de los datos en crudo, los datos preprocesados y la carpeta de resultados
RAW_DATA_FOLDER = relpath("./data/")
PREPO_DATA_FOLDER = relpath("./data_prepo/")
RESULTS_FOLDER = relpath("./results/")
#PLOTS_FOLDER = relpath("./plots/")

# No. de gpus a usar
GPUS_TO_USE = 1 #0 = cpu

# Rutas de los conjunto de datos a preprocesar
PREPO_DS_PATHS = {
   'electricity': relpath("./data_prepo/electricity/"),
   'm1': relpath("./data_prepo/m1/"),
   'm3': relpath("./data_prepo/m3/"),
   'nn5': relpath("./data_prepo/nn5/"),
   'traffic': relpath("./data_prepo/san_francisco_traffic/"),
   #datasets new
   'etth1': relpath("./data_prepo/ett/etth1/"),
   'etth2': relpath("./data_prepo/ett/etth2/"),
   'ettm1': relpath("./data_prepo/ett/etth1/"),
   'ettm2': relpath("./data_prepo/ett/etth2/"),
   'ili': relpath("./data_prepo/ili/"),
   'weather': relpath("./data_prepo/weather/")
}

# Hiperparámetros a probar para el estudio
MODELS_HYPERPARAMS = {
   "rnn": {
      "flavor": ["LSTM", "GRU"],
      "epochs": [100],
      "layers": [1, 2, 4],
      "hidden_units": [32, 128],
      "bs": [16, 32],
      "lr": [1e-3]
   },
   "tcn": {
      "epochs": [200],
      "layers": [4, 1],
      "hidden_size": [64],
      "dropout": [0, 0.1],
      "bs": [16, 32],
      "lr": [1e-2]
   },
}

# MODELS_HYPERPARAMS = {
#    "rnn": {
#       "flavor": ["LSTM", "GRU"],
#       "epochs": [50, 100, 200],
#       "layers": [1, 2, 4],
#       "hidden_units": [32, 64, 128],
#       "bs": [16, 32],
#       "lr": [1e-3, 1e-2]
#    },
#    "tcn": {
#       "epochs": [50, 100, 200],
#       "layers": [1, 2, 4],
#       "hidden_size": [16, 32, 64],
#       "dropout": [0, 0.1, 0.25],
#       "bs": [16, 32],
#       "lr": [1e-3, 1e-2]
#    },
# }

# Número de lags que representan un periodo. Si la seasonality es semanal, 1 lag = 1 día. 7 lags = 1 semana.
SEASONALITY_MAP = {
   "minutely": [1440, 10080, 525960],
   "10_minutes": [144, 1008, 52596],
   "half_hourly": [48, 336, 17532],
   "hourly": 168, #[24, 168, 8766] 1 dia, 1 semana, 1 año
   "daily": 7,
   "weekly": 52,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

FREQUENCY_MAP = {
   "minutely": "1min",
   "10_minutes": "10min",
   "half_hourly": "30min",
   "hourly": "1H",
   "daily": "1D",
   "weekly": "1W",
   "monthly": "1M",
   "quarterly": "1Q",
   "yearly": "1Y"
}
