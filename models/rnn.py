import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from metrics.wape import WeightedAveragePercentageError


class MIMORNNModel(nn.Module):
    def __init__(self, num_rnn_layers: int = 1, hidden_dim: int = 30, input_features: int = 1, input_size: int = 1, output_size: int = 1, dropout=0, return_sequences=False, flavor="LSTM") -> None:
        # Inicializo mi clase padre
        super(MIMORNNModel, self).__init__()
        self.return_sequences = return_sequences
        
        # Construyo la arquitectura en función de mis hiperparámetros
        # Creo la primera capa: Características de entrada = características de un paso temporal
        # Características de salida: Longitud del estado oculto.
        rnn = getattr(nn, flavor.upper())(num_layers=num_rnn_layers, input_size=input_features, hidden_size=hidden_dim, dropout=dropout, batch_first=True)
        self.rnn = rnn
        
        #Aplanamiento
        self.flatten = nn.Flatten()
        # Capa densa (o lineal) para realizar la predicción
        self.relu = nn.ReLU()
        
        dim = hidden_dim
        if return_sequences:
            dim = int(hidden_dim*input_size)
            
        self.linear = nn.Linear(in_features=dim, out_features=dim//2)
        self.linear2 = nn.Linear(in_features=dim//2, out_features=output_size)
        self.init_weights()
        
    def init_weights(self):
        for names in self.rnn._all_weights:
            if isinstance(self.rnn, nn.LSTM):
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1) # Forget a 1, para no olvidar tanto al principio
                
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(self.rnn, name)
                    nn.init.kaiming_normal_(weight.data, mode='fan_in', nonlinearity='relu') 
    # En el método forward, defino cómo interactuan los diferentes componentes del modelo
    # Método forward: Paso hacia adelante, realizar predicciones
    #   x_in: Tensor de entrada
    #   state: En caso de querer inicializar los estados ocultos de la red a un valor
    def forward(self, x_in, state=None):
        # Procesar la entrada en la primera capa en función del tensor de salida y
        # del estado que se use para su inicialización. Las salidas de una capa recurrente son:
        # h: Tensor de dimensiones (lote, longitud secuencia, hidden_dim). Salida de la capa para cada secuencia del lote
        # (h_t, c_t): Conatenación del último estado oculto de cada capa. 
        h, statet = self.rnn(x_in, state) if state is not None else self.rnn(x_in)
   
        #h = h[:, -1, :] # Solo la última salida del modelo (estado oculto con toda la información)
        if self.return_sequences:
            h = self.flatten(h)
        else:
            h = h[:, -1, :] # solo el último estado h
        
        h = self.relu(self.linear(h))
        h = self.linear2(h)

        # Como la entrada posee la forma (BS, LEN, CARACTERISTICAS), ajustamos la salida para
        # que de también esta forma. Como predecimos una única característica, la forma de f_out
        # es (BS, LEN). Explicitamos la existencia de esa característica añadiendo una nueva dimensión
        # de tamaño 1: (BS, LEN, 1)
        # [[A, B]] -> [[[A], [B]]]
        f_out = torch.unsqueeze(h, dim=-1)

        # Devuelvo el resultado
        return f_out, statet
    

class MIMORNN(pl.LightningModule):
  # Hiperparámetros del modelo y del entrenamiento
    def __init__(self, num_rnn_layers: int = 1, hidden_dim: int = 30, input_features: int = 1, input_size: int = 1, output_size: int = 1, lr=1e-3, dropout=0, model_name="mimo_rnn", return_sequences=False, flavor="LSTM", normalizers=[], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Modelo anterior
        self.model = MIMORNNModel(num_rnn_layers=num_rnn_layers, hidden_dim=hidden_dim, input_features=input_features, input_size=input_size, output_size=output_size, flavor=flavor, return_sequences=return_sequences, dropout=dropout)

        # Ratio de aprendizaje del optimizador
        self.lr = lr

        # Función de pérdida
        self.loss = nn.SmoothL1Loss()

        # Horizonte de predicción y tamaño de la entrada
        self.horizonte = output_size
        self.input_size = input_size

        # Contador de épocas
        self.epoch = 0

        # Identificador del modelo
        self.model_name = model_name

        # Registro de la pérdida (loss) por batch y época
        self.train_loss = []
        self.val_loss = []
        self.epoch_train_loss = []
        self.epoch_val_loss = []

        self.normalizers = normalizers
        # Métricas
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mse = torchmetrics.MeanSquaredError(squared=True)
        self.smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        self.wape = WeightedAveragePercentageError()
    # ====== ENTRENAMIENTO =======
    # Al inicio del entrenamiento, limpiar las listas que guardan la pérdida por batch
    def on_train_start(self) -> None:
        self.epoch_train_loss.clear()
        self.train_loss.clear()
        self.epoch_train_loss.clear()
        self.epoch_val_loss.clear()
        return super().on_train_start()

    # Al finalizar cada época del entrenamiento, calcular la pérdida media y registrarla en Tensorboard
    # Registrar también en una misma gráfica la pérdida de entrenamiento y validación
    def on_train_epoch_end(self) -> None:
        epoch_loss = sum(self.train_loss)/len(self.train_loss)
        self.epoch_train_loss.append(epoch_loss)
        self.train_loss.clear()
        self.log_dict({"epoch_train_loss": epoch_loss, "step": self.epoch}, sync_dist=True, prog_bar=True)
        if(len(self.epoch_val_loss) > 0):
            self.logger.experiment.add_scalars('epoch_loss', {'train': epoch_loss, 'val': self.epoch_val_loss[self.epoch]}, global_step=self.epoch, sync_dist=True)

        self.epoch+=1
        return super().on_train_epoch_end()

    # Paso de entrenamiento.
    def training_step(self, batch, batch_idx):
        x, y = batch # Obtener la entrada y etiquetas del batch suministrado por el dataloader
        pred, state = self.model(x) # Realizo la predicción
        loss = self.loss(pred, y) # Calculo la pérdida y la registro
        self.train_loss.append(loss.cpu().detach().numpy())
        return loss

    # ====== VALIDACIÓN =======
    # Al finalizar una época de validación, calcular su pérdida media, guardarla y registrarla
    # en TensorBoard
    def on_validation_epoch_end(self) -> None:
        epoch_loss = sum(self.val_loss)/len(self.val_loss)
        self.epoch_val_loss.append(epoch_loss)
        self.val_loss.clear()
        self.log_dict({"epoch_val_loss": epoch_loss, "step": self.epoch}, sync_dist=True)
        return super().on_validation_epoch_end()

    # Paso de validación. Similar al de entrenamiento.
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred, state = self.model(x)
        loss = self.loss(pred, y)
        self.val_loss.append(loss.cpu().detach().numpy())
        return loss

    # ====== TEST =======
    # Paso de predicción (test).
    def test_step(self, batch, batch_idx):
        x, y = batch # Obtenemos la entrada y las etiquetas esperadas
        pred, state = self.model(x)
        y_np = y.cpu().detach().numpy()
        pred_np = pred.cpu().detach().numpy()
        if len(self.normalizers) > 0:
            y_denorm = []
            pred_denorm = []
            for i in range(y_np.shape[0]):
                y_denorm.append(self.normalizers[i].inverse_transform(y_np[i])) #bs*batch_idx
                pred_denorm.append(self.normalizers[i].inverse_transform(pred_np[i]))
            y = torch.tensor(y_denorm).float()
            pred = torch.tensor(pred_denorm).float()
        pred = pred.reshape((-1,pred.size(dim=1)))
        y = y.reshape((-1, pred.size(dim=1)))
        self.rmse(pred, y)
        self.mse(pred, y)
        self.wape(pred, y)
        self.smape(pred, y)
        loss = self.loss(pred, y) # También calculamos la pérdida
        # Registramos las métricas calculadas.
        self.log_dict({'test_rmse': self.rmse, 'test_mse': self.mse, 'test_wape': self.wape, 'test_smape': self.smape, 'test_loss': loss}, sync_dist=True)

    # ===== PRODUCCION =====
    # Devolver predicciones
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pred, state = self.model(batch)
        return pred.cpu().detach().numpy()
    
    # ===== CONFIGURACIÓN DEL MODELO =====
    # Optimizador a usar
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer