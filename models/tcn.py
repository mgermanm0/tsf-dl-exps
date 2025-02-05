import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from metrics.wape import WeightedAveragePercentageError

class Chomp1d(nn.Module):
    # Eliminar padding usado en las convoluciones causales
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
# Bloque recurrente de la TCN
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)) # Convolución causal 1 con normalización de los pesos
        self.chomp1 = Chomp1d(padding) # Eliminar padding extra
        self.relu1 = nn.ReLU() # Activación
        self.dropout1 = nn.Dropout(dropout) # Dropout

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Bloque recurrente como modelo secuencial
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # En caso de que los canales de entrada no sean iguales a los de salida (para aplicar conexiones residuales)
        # Aplicar una convolucional para que n_inputs = n_outputs (convolución opcional de la conexión residual)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self): # inicializar pesos
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x) # Aplicar convoluciones
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # Salida del bloque: + res = conexión residual.
    
class TemporalConvNetModel(nn.Module):
    '''
      input_feat = número de características de entrada
      tcn_hidden_size = Número de canales (profundida de la matriz de convolución)
      tcn_n_layers = Número de bloques recurrentes que forman la red
      tcn_dropout = % de dropout a aplicar en las convoluciones de los bloques recurrentes
      out_size = Longitud de la secuencia de salida esperada
    '''
    def __init__(self, input_feat, tcn_hidden_size, tcn_n_layers, tcn_dropout, out_size):
        super(TemporalConvNetModel, self).__init__()
        self.input_feat, self.tcn_hidden_size, self.tcn_n_layers, self.tcn_dropout, self.out_size = input_feat,\
                tcn_hidden_size, \
                tcn_n_layers,\
                tcn_dropout,\
                out_size
        # Número de canales estático para cada capa
        num_channels = [tcn_hidden_size] * tcn_n_layers
        kernel_size = 2 # Tamaño de la convolución
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # Factor de dilatación = 2^i, i=numero de bloque (capa)
            in_channels = input_feat if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # one temporalBlock can be seen from fig1(b).
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=tcn_dropout)]

        self.network = nn.Sequential(*layers) # Bloque recurrente
        self.out_proj = nn.Linear(tcn_hidden_size, out_size) # Transformar a la longitud deseada

    def forward(self, x):
        # x: batch_size * seq_len * input_size
        x = x.transpose(1, 2) # x: batch_size * input_size * seq_len
        out = self.network(x)[:, :, -1:] # batch_size * hidden_size * 1
        out = self.out_proj(out.transpose(1, 2)) # transpose (flatten): batch_size * 1 * hidden_size -> out_proy: batch_size * 1 * out_size
        out = out.transpose(2, 1) # batch_size * out_size * 1
        return out
    
class TemporalConvNet(pl.LightningModule):
    def __init__(self, input_feat, tcn_hidden_size, tcn_n_layers, tcn_dropout, out_size, lr=1e-3, normalizers=[], model_name="tcn"):
        super(TemporalConvNet, self).__init__()
        self.model = TemporalConvNetModel(input_feat=input_feat, tcn_hidden_size=tcn_hidden_size, tcn_n_layers=tcn_n_layers, tcn_dropout=tcn_dropout, out_size=out_size)
        self.lr = lr
        self.name = model_name

        # Función de pérdida
        self.loss = nn.SmoothL1Loss()
        # Contador de épocas
        self.epoch = 0
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
    # Paso de entrenamiento.
    def training_step(self, batch, batch_idx):
        x, y = batch # Obtener la entrada y etiquetas del batch suministrado por el dataloader
        pred = self.model(x) # Realizo la predicción
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
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.val_loss.append(loss.cpu().detach().numpy())
        return loss

    # ====== TEST =======
    # Paso de predicción (test).
    def test_step(self, batch, batch_idx):
        x, y = batch # Obtenemos la entrada y las etiquetas esperadas
        pred = self.model(x)
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
        pred = self.model(batch)
        return pred.cpu().detach().numpy()
    
    # ===== CONFIGURACIÓN DEL MODELO =====
    # Optimizador a usar
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer