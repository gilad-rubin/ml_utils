import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import cross_val_score
import itertools


# Custom evaluator class for cross-validation
class CVEvaluator:
    def __init__(self, model, scorer, n_splits):
        self.model = model
        self.n_splits = n_splits
        self.scorer = scorer

    def evaluate(self, X, y):
        scores = cross_val_score(
            self.model, X, y, cv=self.n_splits, scoring=self.scorer, n_jobs=-1
        )
        return np.mean(scores)


# Feature interactions calculator class
class SynergyMatrix:
    def __init__(self, model, evaluator):
        self.model = model
        self.evaluator = evaluator
        self.col_scores = {}
        self.pair_scores = {}
        self.matrix = None

    def _populate_matrix(self):
        cols = list(self.col_scores.keys())
        self.matrix = pd.DataFrame(index=cols, columns=cols, data=0.0)
        for (col1, col2), score in self.pair_scores.items():
            self.matrix.at[col1, col2] = (2 * score) - (
                self.col_scores[col1] + self.col_scores[col2]
            )
            self.matrix.at[col2, col1] = (2 * score) - (
                self.col_scores[col1] + self.col_scores[col2]
            )

    def calculate_synergy_matrix(self, X, y):
        for col in X.columns:
            self.col_scores[col] = self.evaluator.evaluate(X[[col]], y)
        for pair in itertools.combinations(X.columns, 2):
            self.pair_scores[pair] = self.evaluator.evaluate(X[list(pair)], y)
        self._populate_matrix()

    def plot_matrix(self):
        # Cluster the matrix
        linkage_matrix = linkage(
            self.matrix.fillna(0), method="average"
        )  # Use average linkage clustering
        self.dendro = dendrogram(linkage_matrix, no_plot=True)
        order = self.dendro["leaves"]

        sorted_matrix = self.matrix.iloc[order, order]  # Reorder the matrix

        plt.figure(figsize=(20, 18))
        sns.heatmap(
            sorted_matrix, annot=True, cmap="YlOrRd_r", fmt=".2f", annot_kws={"size": 8}
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(
            rotation=0, fontsize=10
        )  # Rotate y labels to horizontal for better readability
        plt.title("Synergy Matrix")
        plt.tight_layout()
        plt.show()
