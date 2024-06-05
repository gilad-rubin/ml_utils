import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from math import log



class Distinguisher:
    def __init__(self, n_bins=5, bin_strategy="quantile", min_recall=0.05, min_precision=0.0, tol=0.01, beam_width=1):
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.tol = tol
        self.beam_width = beam_width

    def fit_transform(self, X):
        self.columns = X.columns
        self.discretizers = {}
        X_filled = X.fillna({col: f"NA_{col}" for col in X.columns})
        for col in X_filled.columns:
            if X_filled[col].dtype == "object" or X_filled[col].dtype.name == "category":
                X_filled[col] = X_filled[col].astype(str)
                self.discretizers[col] = None
            else:
                disc = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.bin_strategy)
                self.discretizers[col] = disc.fit(X_filled[[col]])
        return self.transform(X_filled)

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        X_filled = X.fillna({col: f"NA_{col}" for col in X.columns})
        for col in X_filled.columns:
            if self.discretizers[col] is None:
                X_transformed[col] = X_filled[col]
            else:
                transformed = self.discretizers[col].transform(X_filled[[col]])
                X_transformed[col] = transformed.flatten()
        return X_transformed

    def compute_WoE(self, y):
        p_total = np.sum(y == 1)
        n_total = np.sum(y == 0)
        if p_total == 0 or n_total == 0:
            return 0  # Avoid division by zero
        return log((p_total / (p_total + n_total)) / (n_total / (p_total + n_total)))

    def find_best_rule(self, X, y, min_samples, best_score):
        best_rule = None
        for col in X.columns:
            values = np.unique(X[col])
            thresholds = np.concatenate([values, [max(values) + 1]])
            for val in thresholds:
                for operator in ["<=", ">"]:
                    rule = f"`{col}` {operator} {val}"
                    mask = X.eval(rule)
                    if mask.sum() <= min_samples:
                        continue
                    score = self.compute_WoE(y[mask])
                    if score > best_score:
                        best_score = score
                        best_rule = rule
        return best_rule, best_score

    def get_opposite_rule(self, rule):
        # Assumes rules are of the form '`column` <= value' or '`column` > value'
        if '<=' in rule:
            column, value = rule.split(' <= ')
            return f'{column} > {value}'
        elif '>' in rule:
            column, value = rule.split(' > ')
            return f'{column} <= {value}'
        return None
    
    def generate_rules(self, X, y, current_rule='', rules=[], applied_rules=set(), depth=0, baseline_woe=None):
        if depth > 5 or not X.size:
            return rules
        if baseline_woe is None:
            baseline_woe = self.compute_WoE(y)  # Calculate baseline WoE for the entire dataset at the start
        
        min_samples = int(len(X) * self.min_recall)
        rule, score = self.find_best_rule(X, y, min_samples, baseline_woe)
        rule_mask = X.eval(rule)
        if rule is None or rule in applied_rules:  # Ensure rule is valid and impactful
            return rules

        applied_rules.add(rule)  # Mark this rule as applied
        recall = y[rule_mask].sum() / y.sum()
        precision = y[rule_mask].sum() / rule_mask.sum()
        rule_woe = self.compute_WoE(y[rule_mask])

        if recall >= self.min_recall and precision >= self.min_precision and rule_woe > baseline_woe:
            new_rule = f"({current_rule} & {rule})" if current_rule else rule
            rules.append([new_rule, {'recall': recall, 'precision': precision, 'WoE': rule_woe}])
            # Recurse on both partitions with the updated baseline WoE
            self.generate_rules(X[rule_mask], y[rule_mask], new_rule, rules, applied_rules, depth + 1)
            self.generate_rules(X[~rule_mask], y[~rule_mask], self.get_opposite_rule(rule), rules, applied_rules, depth + 1)

        return rules

    def get_opposite_rule(self, rule):
        # Assumes rules are of the form 'column <= value' or 'column > value'
        if '<=' in rule:
            column, value = rule.split(' <= ')
            return f'{column.strip()} > {value.strip()}'
        elif '>' in rule:
            column, value = rule.split(' > ')
            return f'{column.strip()} <= {value.strip()}'
        return None

# Usage of the class
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = (iris.target == 1).astype(int)  # Binary classification: 1 if 'setosa', 0 otherwise
X.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in X.columns]
df = X.copy()
df["target"] = y.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
distinguisher = Distinguisher(n_bins=5, min_recall=0.01, bin_strategy="quantile")
X_train_transformed = distinguisher.fit_transform(X_train)
rules = distinguisher.generate_rules(X_train, y_train)
print("Generated Rules:")
for rule in rules:
    print(rule)