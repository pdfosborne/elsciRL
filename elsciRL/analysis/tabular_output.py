import numpy as np
import pandas as pd

class TabularOutput:
    def __init__(self, results_data, save_dir):
        self.results_data = results_data
        self.save_dir = save_dir
        self.num_episodes = np.max(results_data['episode'])

    def save_results(self):
        pd.DataFrame(self.results_data).to_csv(self.save_dir+'/results.csv')