from torchmetrics import Metric
import torch
class WeightedAveragePercentageError(Metric):
  def __init__(self):
    super().__init__()
    self.add_state("error", default=torch.tensor(0), dist_reduce_fx="sum")
    self.add_state("total_real", default=torch.tensor(0), dist_reduce_fx="sum")
    
  def update(self, preds: torch.Tensor, target: torch.Tensor):
    error = torch.sum(torch.abs(target - preds))
    sum_total = torch.sum(target)
    
    self.error += error.long()
    self.total_real += sum_total.long()

  def compute(self):
    return self.error / self.total_real