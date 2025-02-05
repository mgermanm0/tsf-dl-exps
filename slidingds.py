import torch
import numpy as np
from torch.utils.data import Dataset

GENERAL_MODEL = 1 # Usar un scaler con todo el conjunto de datos
SERIES_MODEL = 2 # Usar un scaler para cada serie

class SlidingWindowDataset(Dataset):
    """
    Parámetros
      - ds: Dataset completo
      - train_size: Proporción a usar como conjunto de entrenamiento
      - past_history: Historial de lags a usar para la predicción
      - forecast_horizon: Horizonte de predicción (número de instantes temporales a predecir)
      - stride: Separación entre ventanas.
      - targets: Variables a predecir (target)
    """
    def __init__(self, ds, past_history, forecast_horizon, stride=1, cols=[], targets=[], normalizers=[], mode=GENERAL_MODEL) -> None:
        self.past_history = past_history # lags usados para la predicción
        self.forecast_horizon = forecast_horizon # Tamaño del horizonte a predecir
        self.stride = stride # Separación entre ventanas
        self.mode = mode
        self.serieidx = 0 # conocimiento no general
        # Obtener índices de columnas a predecir
        
        del_feat = None
        if len(cols) > 0:
          feat_idx = [cols.get_loc(feat) for feat in targets]
          drop_idx = [i for i in range(len(cols)) if i not in feat_idx]
          del_feat = lambda x: np.apply_along_axis(lambda arr: np.delete(arr, drop_idx), 1, x)

        # Creación de las ventanas
        x, y = [], [] # Conjuntos de entrada y salida (X, Y)
        for n_serie, serie in enumerate(ds): # Para cada serie del dataset
          series_x = []
          series_y = []
          for i in range(past_history, len(serie) - forecast_horizon + 1, self.stride):
            xslice = serie[i-past_history:i].astype(np.float32)
            yslice = serie[i:i+forecast_horizon].astype(np.float32)
            if len(normalizers) >= 0:
              normalizer = None
              if mode == SERIES_MODEL:
                normalizer = normalizers[n_serie]
              else:
                normalizer = normalizers[-1]
              xslice = normalizer.transform(xslice)

            series_x.append(np.array(xslice))
            series_y.append(np.array(del_feat(yslice)) if del_feat else np.array(yslice))
          if mode == GENERAL_MODEL:
            x.extend(np.array(series_x))
            y.extend(np.array(series_y))
          else:
            x.append(np.array(series_x))
            y.append(np.array(series_y))
        self.x = x
        self.y = y

    def __str__(self) -> str:
        out = f'\nSlidingWindowDataset(past_history={self.past_history}, horizon={self.forecast_horizon}, stride={self.stride})\n'
        out += f'SHAPES:\n\tX shape {len(self.x)}: {self.x[-1].shape}\n\tY shape {len(self.y)}: {self.y[-1].shape}\n'
        return out

    # Necesario para que DataLoader sepa el tamaño del dataset
    def __len__(self):
        if self.mode == GENERAL_MODEL:
          return len(self.x)
        
        return self.x[self.serieidx].shape[0]

    # Obtener un elemento del dataset usando el operador corchete [indice]
    def __getitem__(self, index): # Es necesario convertir a tensor para que PyTorch pueda usarlo
      if self.mode == GENERAL_MODEL:
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(self.y[index]).float()
      return torch.from_numpy(self.x[self.serieidx][index]).float(), torch.from_numpy(self.y[self.serieidx][index]).float()
    
    def set_next_serie_idx(self):
      if self.mode == SERIES_MODEL:
        self.serieidx += 1
    
    def have_series(self):
      if self.serieidx < len(self.x):
        return True
      return False
    
    def get_serie_idx(self):
      return self.serieidx
    
    def reset_series_idx(self):
      if self.mode == SERIES_MODEL:
        self.serieidx = 0