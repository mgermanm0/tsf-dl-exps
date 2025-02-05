import pandas as pd
import os
def read_results_file(csv_filepath, colums):
    if os.path.isfile(csv_filepath):
        return pd.read_csv(csv_filepath, sep=";", index_col=0)
    
    results = pd.DataFrame(columns=colums)
    return results