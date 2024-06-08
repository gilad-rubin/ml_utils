import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def mean_scoring_func(x, y):
    return y.groupby(x).mean()


def woe_score(goods, bads, total_goods, total_bads):
    # Check if total counts are zero
    if total_goods == 0:
        raise ValueError("Total goods count cannot be zero.")
    if total_bads == 0:
        raise ValueError("Total bads count cannot be zero.")

    # Avoid calculations if goods are zero
    if goods == 0:
        return 0

    # Handle zero bads by assigning a small value to avoid division by zero
    if bads == 0:
        bads = 1e-10

    # Calculate proportions
    good_proportion = goods / total_goods
    bad_proportion = bads / total_bads

    # Calculate WoE
    woe = np.log(good_proportion / bad_proportion)
    return woe


def woe_scoring_func(x, y):
    total_goods = y.sum()
    if total_goods == 0:
        return None
    total_bads = len(y) - total_goods
    # Explicitly set observed=True to include only categories present in the data
    grouped = y.groupby(x, observed=True).agg(["sum", "count"])
    grouped["goods"] = grouped["sum"]
    grouped["bads"] = grouped["count"] - grouped["goods"]
    woe_scores = grouped.apply(
        lambda row: woe_score(row["goods"], row["bads"], total_goods, total_bads),
        axis=1,
    )
    return woe_scores


class CategoricalSorter:
    def __init__(self, scoring_function):
        self.scoring_function = scoring_function
        self.sorted_categories = {}

    def fit(self, X, y, categorical_columns=None):
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=["category"]).columns

        for col in categorical_columns:
            if X[col].isna().any():
                # Handle NaN values by including them in the scoring process if necessary
                # Here we assume NaNs are handled externally in the scoring function if needed
                self.sorted_categories[col] = [np.nan]
            else:
                self.sorted_categories[col] = []

            category_scores = self.scoring_function(X[col].dropna(), y[X[col].notna()])
            if category_scores is not None:
                sorted_categories = category_scores.sort_values().index.tolist()
            else:
                sorted_categories = X[col].dropna().unique().tolist()

            # Include NaN handling by adding NaN at the end with index -1
            self.sorted_categories[col] = sorted_categories
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            if col in self.sorted_categories:
                category_mapping = {
                    category: idx
                    for idx, category in enumerate(self.sorted_categories[col])
                }
                # Assign -1 to NaN values
                category_mapping[np.nan] = -1
                X_transformed[col] = X_transformed[col].map(category_mapping).fillna(-1)
        return X_transformed


# TODO: consider removing this and using Ordinal Encoder by sklearn
# TODO: then, think how to handle the sorting mechanism and where to put "inverse transform"
class SortedOrdinalEncoder:
    def __init__(self, sorter=None, encoded_missing_value=np.nan):
        self.sorter = sorter
        self.category_mappings = {}  # To store mappings created by the sorter
        self.encoded_missing_value = encoded_missing_value

    def fit(self, X, y):
        if self.sorter:
            self.sorter.fit(X, y)
            # Update the mappings based on the sorter directly
            for col in X.columns:
                if col in self.sorter.sorted_categories:
                    sorted_categories = self.sorter.sorted_categories[col]
                    # Create mapping, ensuring all categories are included and NaN is handled separately
                    category_mapping = {
                        category: idx
                        for idx, category in enumerate(sorted_categories)
                        if category is not np.nan
                    }
                    category_mapping[
                        np.nan
                    ] = self.encoded_missing_value  # Explicit NaN mapping
                    self.category_mappings[col] = category_mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            if col in self.category_mappings:
                # Apply the mappings, and specifically handle NaNs according to encoded_missing_value
                X_transformed[col] = X_transformed[col].map(self.category_mappings[col])
                if self.encoded_missing_value is not np.nan:
                    X_transformed[col].fillna(self.encoded_missing_value, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, encoded_values):
        decoded_frames = {}
        for col in encoded_values.columns:
            if col in self.category_mappings:
                # We need to inverse map while considering the handling of missing values
                inverse_map = {
                    v: k for k, v in self.category_mappings[col].items() if v != -1
                }
                inverse_map[
                    self.encoded_missing_value
                ] = np.nan  # Map encoded_missing_value back to NaN
                decoded_frames[col] = encoded_values[col].map(inverse_map)
        return pd.DataFrame(decoded_frames)


class BinningStrategy:
    def calculate_bins(self, data, max_bins):
        raise NotImplementedError(
            "Each strategy must implement the calculate_bins method."
        )


class FreedmanDiaconisStrategy(BinningStrategy):
    def __init__(self, factor=1.5):
        self.factor = factor

    def calculate_bins(self, data, max_bins):
        data = data.dropna()  # Ignore NaN values in data for bin calculation
        if data.nunique() <= 1:
            return 1  # Only one unique value or all values are NaN

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            return 1  # Return 1 bin if IQR is 0 to avoid division by zero

        # Calculate bin width
        bin_width = self.factor * IQR / (data.size ** (1 / 3))

        if bin_width > 0:
            optimal_bins = int((data.max() - data.min()) / bin_width)
            return min(optimal_bins, max_bins)
        else:
            return 1  # Default to 1 bin if bin width calculation fails


class NBinsCalculator:
    def __init__(self, strategy, max_bins=20):
        self.strategy = strategy
        self.max_bins = max_bins

    def calculate_bins(self, X):
        n_bins = []
        for column in X.columns:
            data = X[column].dropna()  # Remove NaN values for processing
            if data.dtype in ["object", "category"] or pd.api.types.is_string_dtype(
                data
            ):
                # Handle categorical and string data
                unique_values = data.nunique()
                if unique_values == 1:
                    bins = 1
                else:
                    bins = min(unique_values, self.max_bins)
            else:
                # Use the strategy for numerical data
                bins = self.strategy.calculate_bins(data, self.max_bins)
            n_bins.append(bins)
        return n_bins


# TODO: consider switching to sklearn wrapper
class PandasQCutDiscretizer:
    def __init__(self, bin_calculator=None):
        self.bin_calculator = bin_calculator
        self.bins = None
        self.left_edges = {}
        self.right_edges = {}

    def fit(self, X, n_bins=None):
        if self.bin_calculator is not None:
            # Calculate the optimal number of bins for each column using the calculator
            self.bins = self.bin_calculator.calculate_bins(X)
        elif n_bins is not None:
            # Use the provided number of bins
            self.bins = (
                n_bins if isinstance(n_bins, list) else [n_bins] * len(X.columns)
            )
        else:
            raise ValueError(
                "n_bins must be provided if no bin_calculator is specified."
            )
        return self

    def transform(self, X):
        if self.bins is None:
            raise RuntimeError("The discretizer has not been fitted yet.")
        X_transformed = X.copy()
        for idx, col in enumerate(X.columns):
            if X[col].dtype in [
                np.float64,
                np.int64,
            ]:  # Discretize numeric columns only
                X_transformed[col], bins = pd.qcut(
                    X[col], self.bins[idx], retbins=True, duplicates="drop"
                )
                self.left_edges[col] = bins[:-1]
                self.right_edges[col] = bins[1:]
        return X_transformed

    def fit_transform(self, X, n_bins=None):
        self.fit(X, n_bins)
        return self.transform(X)

    def get_bin_edges(self):
        return self.left_edges, self.right_edges


from abc import ABC, abstractmethod


class QueryEvaluator(ABC):
    @abstractmethod
    def to_query(self):
        """Generate a string suitable for use in DataFrame.query()."""
        pass

    def __str__(self):
        return self.to_query()

    def query(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        """Execute the query against a pandas DataFrame."""
        query_string = self.to_query()
        return df.query(query_string)


from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Rule(QueryEvaluator):
    def __init__(self, feature, operator, value, score=None):
        self.feature = feature
        self.operator = operator
        self.value = value if operator not in ["isna", "notnull"] else None
        self.score = score

    def to_query(self):
        if self.operator in ["isna", "notnull"]:
            return f"`{self.feature}`.{self.operator}()"
        elif isinstance(self.value, str):
            return f"`{self.feature}` {self.operator} '{self.value}'"
        else:
            return f"`{self.feature}` {self.operator} {self.value}"

    def negate(self):
        negation_map = {
            "<=": ">",
            ">=": "<",
            "<": ">=",
            ">": "<=",
            "==": "!=",
            "!=": "==",
            "isna": "notnull",
            "notnull": "isna",
        }
        negated_operator = negation_map[self.operator]
        return Rule(self.feature, negated_operator, self.value, self.score)

    def get_mask(self, X):
        if self.operator in ["isna", "notnull"]:
            if self.operator == "isna":
                return X[self.feature].isna()
            else:
                return X[self.feature].notnull()
        else:
            return eval(f"X['{self.feature}'] {self.operator} {self.value}")


class Path(QueryEvaluator):
    def __init__(self, rules=None, score=None):
        self.rules = rules if rules else []
        self.score = score  # This should be an instance of RuleScore

    def add_rule(self, rule):
        self.rules.append(rule)

    def to_query(self):
        return " and ".join([f"({rule.to_query()})" for rule in self.rules])

    def get_mask(self, X):
        if not self.rules:
            return np.ones(len(X), dtype=bool)
        mask = self.rules[0].get_mask(X)
        for rule in self.rules[1:]:
            mask &= rule.get_mask(X)
        return mask

    def __str__(self):
        return self.to_query()

    def __repr__(self):
        return self.to_query()

    @staticmethod
    def find_best_path(paths):
        """Finds the best path from a list based on their scores."""
        if not paths:
            return None
        best_path = paths[0]
        for current_path in paths[1:]:
            # print(current_path, current_path.score.scores[0])
            if current_path.score.scores[0] > best_path.score.scores[0]:  # TODO: fix
                # print(f"Found better path: {current_path} vs. {best_path}")
                best_path = current_path
        return best_path


from abc import ABC, abstractmethod
import numpy as np

from abc import ABC, abstractmethod

class ScoreStrategy(ABC):
    name = None
    greater_is_better = True

    @abstractmethod
    def calculate(self, **kwargs):
        """Method to calculate the score, to be implemented by each subclass."""
        pass

    def safe_divide(self, numerator, denominator, default=0):
        """Safely divide two numbers, returning `default` if the denominator is zero."""
        if denominator == 0:
            return default
        return numerator / denominator



class RecallScore(ScoreStrategy):
    name = "recall"
    greater_is_better = True

    def calculate(self, **kwargs):
        true_positives = kwargs.get("true_positives", 0)
        total_positives = kwargs.get("total_positives", 0)
        return true_positives / total_positives if total_positives != 0 else 0


class PrecisionScore(ScoreStrategy):
    name = "precision"
    greater_is_better = True

    def calculate(self, **kwargs):
        true_positives = kwargs.get("true_positives", 0)
        predicted_positives = kwargs.get("predicted_positives", 0)
        return true_positives / predicted_positives if predicted_positives != 0 else 0

class WoEScore(ScoreStrategy):
    name = "WoE"
    greater_is_better = True

    def calculate(self, **kwargs):
        true_positives = kwargs.get('true_positives', 0)
        false_positives = kwargs.get('false_positives', 0)
        total_positives = kwargs.get('total_positives', 0)
        total_negatives = kwargs.get('total_negatives', 0)

        if total_positives == 0 or total_negatives == 0:
            return 0  # Return zero or another indicative value when there's no valid data to process.

        good_ratio = self.safe_divide(true_positives, total_positives, default=0)
        bad_ratio = self.safe_divide(false_positives, total_negatives, default=0)

        if good_ratio <= 0 or bad_ratio <= 0:
            return float('-inf')  # Logarithm of zero or negative is undefined, handle gracefully.

        return np.log(good_ratio / bad_ratio) if good_ratio > 0 and bad_ratio > 0 else float('-inf')


class RuleScore:
    def __init__(self, scores):
        self.scores = scores

    def __eq__(self, other):
        if not isinstance(other, RuleScore):
            return NotImplemented
        return self.scores == other.scores

    @staticmethod
    def is_better(
        score1, score2
    ):  # TODO: add secondary scores etc... incorporate greater_is_better
        """Compare two RuleScores based on a defined order of importance of the scores."""
        if score2 is None:
            return True
        if score1.scores[0] > score2.scores[0]:
            return True

    def __repr__(self):
        return " ".join([str(score) for score in self.scores])  # TODO: fix


class BinaryRuleEvaluator:
    def __init__(self, score_strategies):
        self.score_strategies = score_strategies

    def evaluate(self, rule, X, y):
        mask = rule.get_mask(X)
        true_positives = y[mask].sum()
        total_positives = y.sum()
        predicted_positives = mask.sum()
        total_negatives = len(y) - total_positives
        false_positives = predicted_positives - true_positives

        score_results = []
        for score_strategy in self.score_strategies:
            score_value = score_strategy.calculate(
                true_positives=true_positives,
                false_positives=false_positives,
                total_positives=total_positives,
                total_negatives=total_negatives,
                predicted_positives=predicted_positives,
            )
            score_results.append(score_value)

        return RuleScore(scores=score_results)


class BinaryRuleFilter:
    def __init__(self, min_recall=0.0, max_fpr=0.0):
        """
        Initialize the filter with minimum recall and maximum False Positive Rate (FPR).
        Args:
        min_recall (float or int): Minimum acceptable recall (fraction) or minimum number of true positives (absolute).
        max_fpr (float or int): Maximum acceptable False Positive Rate (fraction) or maximum number of false positives (absolute).
        """
        self.min_recall = min_recall
        self.max_fpr = max_fpr

    def apply(self, rule):
        recall = rule.score.scores[1]  # TODO: fix
        # fpr = score.scores[2] #TODO: fix
        # Determine if recall meets the minimum threshold
        recall_ok = recall >= self.min_recall

        # if isinstance(self.min_recall, float):
        #     recall_ok = recall >= self.min_recall
        # else:
        #     recall_ok = true_positives >= self.min_recall

        # # Determine if FPR meets the maximum threshold
        # if isinstance(self.max_fpr, float):
        #     fpr_ok = fpr <= self.max_fpr
        # else:
        #     fpr_ok = false_positives <= self.max_fpr

        return recall_ok  # and fpr_ok


# distinguisher.plot_rules_tree(chosen_rules)
# TODO: make the default decisions for Distinguisher()
# TODO: support multi-class (One VS Rest)
# TODO: support regression (By Binning Target)
# TODO: rethink classes and calls
import pandas as pd


class PrismRules:
    def __init__(self, encoder, discretizer, evaluator):
        self.encoder = encoder
        self.discretizer = discretizer
        self.evaluator = evaluator
        self.rules = []

    def generate_all_potential_rules(self, X):
        rules = []
        for feature in X.columns:
            left_edges = self.discretizer.left_edges.get(feature, [])
            right_edges = self.discretizer.right_edges.get(feature, [])
            # TODO: test if this makes sense (or redundant)
            rules.extend([Rule(feature, ">", edge) for edge in left_edges])
            rules.extend([Rule(feature, ">=", edge) for edge in left_edges])
            rules.extend([Rule(feature, "<=", edge) for edge in right_edges])
            rules.extend([Rule(feature, "<", edge) for edge in right_edges])

            if X[feature].isna().any():
                rules.append(Rule(feature, "isna", None))

        return rules

    def check_input(self, X, y, cat_cols=None):
        # Validate X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        # Validate y is a numpy array or pandas Series
        y_array = np.asarray(y)
        if len(np.unique(y_array)) != 2:
            raise ValueError("y must have exactly two unique values.")

        # Validate columns in X for data types
        if any(X[col].nunique(dropna=False) <= 1 for col in X.columns):
            raise ValueError(
                "All columns in X must have more than one unique value including NA."
            )

        # Check for data diversity and types
        problematic_cols = {}
        for col in X.columns:
            if X[col].nunique(dropna=False) <= 1 or not (
                pd.api.types.is_numeric_dtype(X[col])
                or pd.api.types.is_categorical_dtype(X[col])
                or col in cat_cols
            ):
                problematic_cols[col] = X[col].dtype

        if problematic_cols:
            error_message = "The following columns do not meet the data type or diversity requirements:\n"
            for col, dtype in problematic_cols.items():
                error_message += f"{col}: {dtype}, expected numeric, categorical, or specified as categorical in cat_cols\n"
            raise ValueError(error_message)

    def find_rules(self, X, y, rule_filter, max_depth=3, cat_cols=None):
        """
        Finds the best set of rules to classify the given data using the Prism algorithm.

        Args:
            X (pandas.DataFrame): The input data to be classified.
            y (pandas.Series): The target variable to be predicted.
            rule_filter (function, optional): A function that filters the potential rules based on some criteria. Defaults to None.
            max_depth (int, optional): The maximum depth of the rule tree. Defaults to 3.
            cat_cols (list, optional): A list of column names representing categorical features. Defaults to None.

        Returns:
            list: The final set of rules that provide the best classification performance.
        """
        informative_cols = [col for col in X.columns if len(pd.unique(X[col])) > 1]

        X_encoded = self.encoder.fit_transform(X[informative_cols], y)
        X_preprocessed = self.discretizer.fit_transform(X_encoded)

        self.all_potential_rules = self.generate_all_potential_rules(X_preprocessed)

        final_rules = self.find_rules_recursive(
            X_encoded,
            y,
            chosen_rules=[],
            applied_path=None,
            rule_filter=rule_filter,
            max_depth=max_depth,
        )

        # self.beautify_rules(final_rules, self.encoder, X)
        return final_rules

    # TODO: rename ?
    # TODO: add self.max_depth? or max_depth here?
    # TODO: enable the building of multiple nodes under the root where it's implicit that it's relevant to the negative of all the rules that have been applied so far
    def find_rules_recursive(
        self, X, y, chosen_rules, applied_path, rule_filter, max_depth=3
    ):
        if applied_path is not None:
            if len(applied_path.rules) >= max_depth:
                return chosen_rules
            
        #print(f"Evaluating with applied path: {applied_path}")
        applied_path = applied_path if applied_path else Path(rules=[], score=None)
        evaluated_paths = []
        for rule in self.all_potential_rules:
            if rule in applied_path.rules:
                continue

            mask = rule.get_mask(X)
            if mask.sum() == len(X) or mask.sum() == applied_path.get_mask(X).sum():
                continue
            path = Path(applied_path.rules + [rule])
            path.score = self.evaluator.evaluate(path, X, y)
            if rule_filter.apply(path):
                evaluated_paths.append((path))

        if len(evaluated_paths) == 0:
            return chosen_rules

        best_path = Path.find_best_path(evaluated_paths)
        #print(f"Best path: {best_path} with score: {best_path.score}")
        #print(mask.sum(), applied_path.get_mask(X).sum(), len(X))
        if applied_path is None or RuleScore.is_better(
            best_path.score, applied_path.score
        ):
            #print(f"Found better path: {best_path}")
            # TODO: understand why there are duplicates here and cancel them
            if best_path in chosen_rules:
                return chosen_rules

            chosen_rules.append(best_path)

            self.find_rules_recursive(
                X, y, chosen_rules, best_path, rule_filter, max_depth
            )

            last_rule = best_path.rules[-1]
            neg_last_rule = last_rule.negate()
            applied_rules = applied_path.rules if applied_path else []
            neg_path = Path(applied_rules + [neg_last_rule])
            neg_path.score = self.evaluator.evaluate(neg_path, X, y)
            #print(f"Negated path: {neg_path} with score: {neg_path.score}")
            #if rule_filter.apply(neg_path):
            #if neg_path in chosen_rules:
            #    return chosen_rules

            #chosen_rules.append(neg_path)
            self.find_rules_recursive(
                X, y, chosen_rules, neg_path, rule_filter, max_depth
            )

        return chosen_rules

    # TODO: see if there is only one value in the rule and turn into x==4 or x!=4
    # TODO: convert "<=0" to ==0 where it's relevant. be aware of floats.
    # TODO: refactor
    def beautify_rules(self, paths, encoder, X):
        non_dup_rules = []
        for path in paths:
            for rule in path.rules:
                if rule not in non_dup_rules:
                    non_dup_rules.append(rule)

        for rule in non_dup_rules:
            try:
                if rule.operator not in ["isna", "notnull"]:
                    if rule.feature in encoder.category_mappings:
                        value = encoder.convert_to_categorical(
                            rule.feature, rule.value, rule.operator == ">"
                        )

                        value_opposite = encoder.convert_to_categorical(
                            rule.feature, rule.value, rule.operator != ">"
                        )

                        if len(value) <= len(value_opposite):
                            rule.value = value
                            rule.operator = "in"
                        else:
                            rule.value = value_opposite
                            rule.operator = "not in"
            except:
                print(f"Failed to beautify rule: {rule}", rule.value)

    # TODO: refactor.
    # TODO: consider this outside of the class? in general - the whole plotting...
    # TODO: add a way to copy-paste the query directly from the node (?)
    def add_nodes_edges(
        self,
        dot,
        node_dict,
        parent_id="root",
        parent_label="Root",
        depth=0,
        idx_in_path=0,
    ):
        for i, (query, data) in enumerate(node_dict.items()):
            current_id = f"{parent_id}_{i}_{depth}_{idx_in_path}"

            # Create the current node
            node = data["_node"]
            # label = f"{str(node)}\nEvent Count: {node.event_count}\nNon-Event Count: {node.non_event_count}\nScore: {node.get_score():.2f}"
            label = f"{str(node)}\nScore:{node.score}"
            dot.node(
                current_id,
                label=label,
                fontsize="12",
                fontname="Helvetica",
                shape="box",
                rounded="true",
                fillcolor="lightblue",
                style="filled, rounded",
            )

            # Connect the current node to its parent
            dot.edge(parent_id, current_id)

            # Move to the children
            self.add_nodes_edges(
                dot, data["_children"], current_id, label, depth + 1, i
            )

    def build_shared_tree(self, paths):
        root = {}
        for path in paths:
            current_dict = root
            for rule in path.rules:
                query = str(rule)
                if query not in current_dict:
                    current_dict[query] = {"_node": rule, "_children": {}}
                current_dict = current_dict[query]["_children"]
        return root

    def plot_tree(self, paths):
        dot = Digraph(
            node_attr={
                "style": "filled",
                "shape": "box",
                "rounded": "true",
                "fillcolor": "lightyellow",
            },
            edge_attr={"color": "black", "arrowsize": "0.6"},
            graph_attr={"fontsize": "16", "fontname": "Helvetica"},
        )

        dot.node("root", "Root", fontcolor="black", fontsize="20", fontname="Helvetica")
        shared_tree = self.build_shared_tree(paths)
        self.add_nodes_edges(dot, shared_tree, "root")
        display(dot)
