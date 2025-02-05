from .transform import Transform
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# Not invertible
class YanOutliersTransform(Transform):
    def __init__(self, median_lags, name="YanOutliersDetection"):
        super().__init__()
        self.name = name
        self.median_lags = median_lags

    def transform(self, series, index_feature=0):
        listts = []
        for i in range(len(series)):
            val = series[i][index_feature]

            if i >= self.median_lags and i+self.median_lags+1 < len(series):
                left = series[i-self.median_lags:i]
                right = series[i+1:i+self.median_lags+1]
                
                ml = np.median(left.reshape(-1))
                mr = np.median(right.reshape(-1))
            
                check = 4*max(abs(ml), abs(mr))
                
                if abs(val) >= check:
                    val = (series[i-1] + series[i+1])/2.0
            
            listts.append(val)
        return np.array(listts, dtype=object).reshape(-1, 1)