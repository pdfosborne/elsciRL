import os
import json

class ResultManager:
    """Handles saving, loading, and reporting of results."""
    def __init__(self, analysis):
        self.analysis = analysis

    def save_results(self, results, save_dir, filename):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        results.to_csv(path)

    def load_results(self, path):
        # Assumes CSV for now
        import pandas as pd
        return pd.read_csv(path)

    def train_report(self, training_results, save_dir, show_figures):
        return self.analysis.train_report(training_results, save_dir, show_figures)

    def test_report(self, testing_results, save_dir, show_figures):
        return self.analysis.test_report(testing_results, save_dir, show_figures)

    def training_variance_report(self, save_dir, show_figures):
        return self.analysis.training_variance_report(save_dir, show_figures)

    def testing_variance_report(self, save_dir, show_figures):
        return self.analysis.testing_variance_report(save_dir, show_figures)
