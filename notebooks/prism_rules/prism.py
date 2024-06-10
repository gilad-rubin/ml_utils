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
                    category_mapping[np.nan] = (
                        self.encoded_missing_value
                    )  # Explicit NaN mapping
                    self.category_mappings[col] = category_mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            if col in self.category_mappings:
                # Apply the mappings, and specifically handle NaNs according to encoded_missing_value
                X_transformed[col] = (
                    X_transformed[col].map(self.category_mappings[col]).astype(float)
                )
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
                inverse_map[self.encoded_missing_value] = (
                    np.nan
                )  # Map encoded_missing_value back to NaN
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
            # if X[col].dtype in [
            #     np.float64,
            #     np.int64,
            # ]:  # Discretize numeric columns only
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

import numpy as np
import pandas as pd


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
        true_positives = kwargs.get("true_positives", 0)
        false_positives = kwargs.get("false_positives", 0)
        total_positives = kwargs.get("total_positives", 0)
        total_negatives = kwargs.get("total_negatives", 0)

        if total_positives == 0 or total_negatives == 0:
            return 0  # Return zero or another indicative value when there's no valid data to process.
        if false_positives == 0:
            false_positives = 1e-10

        good_ratio = self.safe_divide(true_positives, total_positives, default=0)
        bad_ratio = self.safe_divide(false_positives, total_negatives, default=0)

        if good_ratio <= 0 or bad_ratio <= 0:
            return float(
                "-inf"
            )  # Logarithm of zero or negative is undefined, handle gracefully.

        return (
            np.log(good_ratio / bad_ratio)
            if good_ratio > 0 and bad_ratio > 0
            else float("-inf")
        )


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
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import KBinsDiscretizer


class PrismRules:
    def __init__(self, encoder, discretizer, evaluator):
        self.encoder = encoder
        self.discretizer = discretizer
        self.evaluator = evaluator

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
        self.check_input(X, y, cat_cols)
        informative_cols = [col for col in X.columns if len(pd.unique(X[col])) > 1]

        X_encoded = self.encoder.fit_transform(X[informative_cols], y)
        X_sparse

        print("Starting to find rules...")
        start_time = time.time()
        self.rules = self.find_rules_recursive(
            chosen_rules=[],
            applied_path=None,
            rule_filter=rule_filter,
            max_depth=max_depth,
        )
        end_time = time.time()
        print(f"Found {len(self.rules)} rules in {end_time - start_time} seconds")

        final_rules = self.beautify_rules(self.rules, self.encoder.category_mappings)
        return final_rules

    # TODO: rename ?
    # TODO: add self.max_depth? or max_depth here?
    # TODO: enable the building of multiple nodes under the root where it's implicit that it's relevant to the negative of all the rules that have been applied so far
    def find_rules_recursive(
        self, chosen_rules, applied_path, rule_filter, max_depth=3
    ):
        if applied_path is not None:
            if len(applied_path.rules) >= max_depth:
                return chosen_rules

        # print(f"Evaluating with applied path: {applied_path}")
        applied_path = applied_path if applied_path else Path(rules=[], score=None)
        evaluated_paths = []
        start_time = time.time()
        for rule in self.all_potential_rules:
            if rule in applied_path.rules:
                continue
            path = Path(applied_path.rules + [rule])

            path.score = self.evaluator.evaluate(path, self.X, self.y)
            if rule_filter.apply(path):
                evaluated_paths.append((path))
        end_time = time.time()
        print(
            f"Evaluated {len(evaluated_paths)} paths in {end_time - start_time} seconds"
        )

        if len(evaluated_paths) == 0:
            return chosen_rules

        best_path = Path.find_best_path(evaluated_paths)
        # print(f"Best path: {best_path} with score: {best_path.score}")
        # print(mask.sum(), applied_path.get_mask(X).sum(), len(X))
        if applied_path is None or RuleScore.is_better(
            best_path.score, applied_path.score
        ):
            # TODO: understand why there are duplicates here and cancel them
            if best_path in chosen_rules:
                return chosen_rules

            chosen_rules.append(best_path)
            print(f"Path: {best_path} with score: {best_path.score}")

            self.find_rules_recursive(chosen_rules, best_path, rule_filter, max_depth)

            last_rule = best_path.rules[-1]
            neg_last_rule = last_rule.negate()
            applied_rules = applied_path.rules if applied_path else []
            neg_path = Path(applied_rules + [neg_last_rule])
            neg_path.score = self.evaluator.evaluate(neg_path, X, y)

            self.find_rules_recursive(
                X, y, chosen_rules, neg_path, rule_filter, max_depth + 1
            )

        return chosen_rules

    def convert_rule_to_categorical(self, rule, categories):
        # Handle special operators that do not require conversion.
        if rule.operator in ["isna", "notnull", "==", "!="]:
            return rule

        categories = [cat for cat in categories if isinstance(cat, str)]

        # Convert the rule's value from an index to the corresponding category.
        index = int(rule.value)  # Assuming value is already the correct integer index
        if index >= len(categories):
            print("Problem with index", index, "len(categories)", len(categories))
            index = len(categories) - 1  # Clamp index to the last category
        elif index < 0:
            print("Problem with index", index, "len(categories)", len(categories))
            index = 0  # Ensure index is non-negative

        rule_cats = []
        opposite_rule_cats = []
        # Convert index-based comparison to category-based comparison.
        if rule.operator == "<":
            rule_cats = categories[:index]
            opposite_rule_cats = categories[index:]
        elif rule.operator == "<=":
            rule_cats = categories[: index + 1]
            opposite_rule_cats = categories[index + 1 :]
        elif rule.operator == ">":
            rule_cats = categories[index + 1 :]
            opposite_rule_cats = categories[: index + 1]
        elif rule.operator == ">=":
            rule_cats = categories[index:]
            opposite_rule_cats = categories[:index]

        # Decide whether to use 'in' or 'not in' based on which list is shorter
        if len(opposite_rule_cats) < len(rule_cats):
            value = opposite_rule_cats
            operator = "not in"
        else:
            value = rule_cats
            operator = "in"

        # Return the new rule with the chosen operator and value
        return Rule(rule.feature, operator, value)

    def beautify_rules(self, paths, category_mappings):
        new_paths = []
        for path in paths:
            path_rules = []
            for rule in path.rules:
                if rule.feature in category_mappings:
                    new_rule = self.convert_rule_to_categorical(
                        rule, list(category_mappings[rule.feature].keys())
                    )
                    new_rule.score = rule.score
                    path_rules.append(new_rule)
                else:
                    path_rules.append(rule)
            new_path = Path(rules=path_rules, score=path.score)
            new_paths.append(new_path)
        return new_paths

    def beautify_rules3(self, paths):  # , category_mappings, X):
        compact_paths = []
        for path in paths:
            features = [rule.feature for rule in path.rules]
            dup_features = [
                feature for feature in features if features.count(feature) > 1
            ]
            # final_rules = #non dup features:
            final_rules = [
                rule for rule in path.rules if rule.feature not in dup_features
            ]
            for feature in dup_features:
                feature_rules = [rule for rule in path.rules if rule.feature == feature]
                combined_rule = self.combine_rules(feature_rules)
                final_rules.append(combined_rule)

            compact_path = Path(rules=final_rules, score=path.score)
            # Create a new Path with combined rules and the same score
            compact_paths.append(compact_path)
        for path in compact_paths:
            print(path.rules)
        return compact_paths

    def combine_rules(self, rules):
        if not rules:
            return None

        # Sort and filter rules for combination
        rules.sort(key=lambda r: (r.operator, r.value))
        min_rule = max_rule = None

        for rule in rules:
            if rule.operator in ["<", "<="] and (
                not min_rule or rule.value < min_rule.value
            ):
                max_rule = rule
            elif rule.operator in [">", ">="] and (
                not max_rule or rule.value > max_rule.value
            ):
                min_rule = rule
        print(
            "min_rule",
            min_rule,
            "max_rule",
            max_rule,
            "min_rule.value",
            min_rule.value,
            "max_rule.value",
            max_rule.value,
        )
        # Check for combinable conditions
        if min_rule and max_rule:
            if (
                min_rule.value == max_rule.value
                and max_rule.operator == "<="
                and min_rule.operator == ">="
            ):
                return Rule(
                    feature=min_rule.feature, operator="==", value=min_rule.value
                )
            if (
                min_rule.operator in ["<", "<="]
                and max_rule.operator in [">=", ">"]
                and min_rule.value <= max_rule.value
            ):
                return Rule(
                    feature=min_rule.feature,
                    operator=min_rule.operator + max_rule.operator,
                    value=(min_rule.value, max_rule.value),
                )
            if (
                min_rule.operator == "<"
                and max_rule.operator == ">="
                and min_rule.value <= max_rule.value
            ):
                return (
                    None  # Uncombinable rules, return None or original rules as needed
                )
        # Default return the original rules if no combination is possible
        return rules

    # TODO: see if there is only one value in the rule and turn into x==4 or x!=4
    # TODO: convert "<=0" to ==0 where it's relevant. be aware of floats.
    # TODO: refactor
    def beautify_rules2(self, paths, encoder, X):
        # compact rules
        # for path in paths:
        # if there are two rules with the same feature, we can combine them: (< or <=) with (> or >=)
        # so we combine a < 5 and a >=3 to 3 <= a < 5
        # if we have a <=5 and a>=5 we can combine them to a==5

        for rule in paths:
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


if __name__ == "__main__":
    df = pd.read_csv("data/AmesHousing.csv")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    k = 1000
    raw_y = df["SalePrice"].copy()
    X = df.drop("SalePrice", axis=1).copy()
    X = X.head(k)
    raw_y = raw_y.loc[X.index].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    y = (raw_y >= raw_y.quantile(0.5)).astype(int).copy()
    X[cat_cols] = X[cat_cols].astype("category")

    sorter = CategoricalSorter(scoring_function=woe_scoring_func)
    encoder = SortedOrdinalEncoder(sorter)
    bin_calculator = NBinsCalculator(strategy=FreedmanDiaconisStrategy(), max_bins=20)
    discretizer = PandasQCutDiscretizer(bin_calculator=bin_calculator)
    evaluator = BinaryRuleEvaluator(
        score_strategies=[WoEScore(), RecallScore(), PrecisionScore()]
    )
    rule_filter = BinaryRuleFilter(min_recall=0.1)
    prism = PrismRules(encoder, discretizer, evaluator)
    rules = prism.find_rules(X, y, rule_filter, max_depth=3)


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


def discretize_and_create_sparse_matrix(df, num_quantiles=10):
    # df = df.select_dtypes(include=[np.number]).copy()
    sparse_matrix_list = []
    rule_map = {}

    for column in df.columns:
        series = df[column]  # Work with non-NA values to determine bins
        unique_values = series.unique()

        # Check if there are enough unique values to support the desired number of quantiles
        if len(unique_values) < num_quantiles:
            actual_quantiles = len(
                unique_values
            )  # Use the number of unique values as the number of quantiles
        else:
            actual_quantiles = num_quantiles

        has_nans = df[column].isna().any()
        if has_nans:
            nan_column_sparse = csr_matrix(
                df[column].isna().astype(int).values.reshape(-1, 1)
            )
            sparse_matrix_list.append(nan_column_sparse)
            rule_map[f"{column}.isna()"] = len(sparse_matrix_list) - 1

        if (
            actual_quantiles > 1
        ):  # Proceed only if there are at least two bins to create
            discretized, bins = pd.qcut(
                series, q=actual_quantiles, duplicates="drop", retbins=True
            )
            # df[column] = pd.qcut(df[column], q=actual_quantiles, labels=False, duplicates='drop')

            for i in range(len(bins) - 1):
                # print(bins, df[column])
                # <= and > operators for each bin boundary
                in_bin_le = pd.Series(
                    np.where(series.isna(), False, series <= bins[i + 1])
                )
                sparse_matrix_list.append(
                    csr_matrix(in_bin_le.astype(int).values.reshape(-1, 1))
                )
                rule_map[f"`{column}` <= {bins[i+1]}"] = len(sparse_matrix_list) - 1
                # print(f"`{column}` <= {bins[i+1]}", in_bin_le.sum())
                in_bin_gt = pd.Series(
                    np.where(series.isna(), False, series > bins[i + 1])
                )
                sparse_matrix_list.append(
                    csr_matrix(in_bin_gt.astype(int).values.reshape(-1, 1))
                )
                rule_map[f"`{column}` > {bins[i+1]}"] = len(sparse_matrix_list) - 1
                # print(f"`{column}` > {bins[i+1]}", in_bin_gt.sum())
                if i > 0:
                    in_bin_lt = pd.Series(
                        np.where(series.isna(), False, series < bins[i])
                    )
                    sparse_matrix_list.append(
                        csr_matrix(in_bin_lt.astype(int).values.reshape(-1, 1))
                    )
                    rule_map[f"`{column}` < {bins[i]}"] = len(sparse_matrix_list) - 1

                    in_bin_ge = pd.Series(
                        np.where(series.isna(), False, series >= bins[i])
                    )
                    sparse_matrix_list.append(
                        csr_matrix(in_bin_ge.astype(int).values.reshape(-1, 1))
                    )
                    rule_map[f"`{column}` >= {bins[i]}"] = len(sparse_matrix_list) - 1
        else:
            print(f"Not enough unique values in {column} to discretize as requested.")

    final_sparse_matrix = (
        hstack(sparse_matrix_list) if sparse_matrix_list else csr_matrix((len(df), 0))
    )
    return final_sparse_matrix, rule_map


def combine_rules_and_calculate_woe(X, y, rule_map, rules):
    indices = [rule_map[rule] for rule in rules if rule in rule_map]
    if not indices:
        return np.nan  # Return NaN if no valid rules found

    # Initialize the combined rule with the first rule's data
    combined_rule = X[:, indices[0]].toarray()

    # Multiply subsequent rule columns to apply logical AND
    for idx in indices[1:]:
        combined_rule *= X[:, idx].toarray()

    # Calculate WoE
    combined_rule = combined_rule.ravel()  # Flatten the array to 1D
    total_goods = np.sum(y == 1)
    total_bads = np.sum(y == 0)
    goods = np.sum(combined_rule[y == 1])
    bads = np.sum(combined_rule[y == 0])

    if goods == 0 or bads == 0:
        return np.nan  # Avoid division by zero or log(0)

    good_ratio = goods / total_goods
    bad_ratio = bads / total_bads
    woe = np.log(good_ratio / bad_ratio)
    return woe


from scipy.sparse import csr_matrix


def calculate_support_and_goods(X_sparse, y):
    if isinstance(y, pd.Series):
        y = y.values
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Ensure y is a column vector
    y_sparse = csr_matrix(y)  # Convert y to a sparse matrix
    goods = X_sparse.multiply(y_sparse).sum(axis=0)
    support = X_sparse.sum(axis=0)
    return np.array(support).flatten(), np.array(goods).flatten()


def calculate_woe(support, goods, total_goods, total_bads):
    # Adding a small constant to avoid division by zero or log of zero
    if total_goods == 0 or total_bads == 0:
        return -np.inf
    epsilon = 1
    bads = support - goods
    distribution_of_goods = (goods + epsilon) / total_goods
    distribution_of_bads = (bads + epsilon) / total_bads
    woe = np.log(distribution_of_goods / distribution_of_bads)
    return woe


def find_best_rule(X_sparse, y, rule_map, total_goods, total_bads):
    support, goods = calculate_support_and_goods(X_sparse, y)
    woe_scores = calculate_woe(support, goods, total_goods, total_bads)
    best_index = np.argmax(woe_scores)
    best_rule = list(rule_map.keys())[best_index]
    return best_index, best_rule


def filter_sparse_matrix(X_sparse, y, rule_index):
    selector = X_sparse[:, rule_index].toarray().flatten() == 1
    X_filtered = X_sparse[selector]
    y_filtered = y[selector]
    return X_filtered, y_filtered


def iterative_rule_refinement(X_sparse, y, rule_map, min_pos=5):
    total_goods = np.sum(y)
    total_bads = len(y) - total_goods
    previous_best_score = -np.inf

    while X_sparse.get_shape()[1] > 0:
        best_index, best_rule = find_best_rule(
            X_sparse, y, rule_map, total_goods, total_bads
        )
        current_best_score = calculate_woe(
            X_sparse[:, best_index].sum(),
            y[X_sparse[:, best_index].toarray().flatten() == 1].sum(),
            total_goods,
            total_bads,
        )
        if current_best_score <= previous_best_score:  # Stop if no improvement
            print("No improvement in WoE score. Stopping refinement.")
            break

        previous_best_score = current_best_score
        print(f"Refining rule: {best_rule} with WoE score: {current_best_score:.3f}")

        X_sparse, y = filter_sparse_matrix(X_sparse, y, best_index)

        if X_sparse.get_shape()[1] == 0:
            break

        # Remove columns with fewer than min_pos positive entries
        column_sums = X_sparse.sum(axis=0).A.flatten()
        valid_columns = column_sums >= min_pos
        X_sparse = X_sparse[:, valid_columns]

        # Rebuild rule_map with new valid column indices
        old_to_new_indices = {
            old: new for new, old in enumerate(np.where(valid_columns)[0])
        }
        rule_map = {
            rule: old_to_new_indices[idx]
            for rule, idx in rule_map.items()
            if idx in old_to_new_indices
        }


def find_best_rule_in_X(
    X_sparse, y, rule_map, total_goods, total_bads, previous_best_score, min_pos=5
):
    best_index, best_rule = find_best_rule(
        X_sparse, y, rule_map, total_goods, total_bads
    )
    current_best_score = calculate_woe(
        X_sparse[:, best_index].sum(),
        y[X_sparse[:, best_index].toarray().flatten() == 1].sum(),
        total_goods,
        total_bads,
    )
    if previous_best_score >= current_best_score:  # Stop if no improvement
        print("No improvement in WoE score. Stopping refinement.")
        return None, None, None, None, None, None

    X_sparse_filtered, y_filtered = filter_sparse_matrix(X_sparse, y, best_index)

    # Remove columns with fewer than min_pos positive entries
    column_sums = X_sparse_filtered.sum(axis=0).A.flatten()
    valid_columns = column_sums >= min_pos
    X_sparse_filtered = X_sparse_filtered[:, valid_columns]

    # Rebuild rule_map with new valid column indices
    old_to_new_indices = {
        old: new for new, old in enumerate(np.where(valid_columns)[0])
    }
    new_rule_map = {
        rule: old_to_new_indices[idx]
        for rule, idx in rule_map.items()
        if idx in old_to_new_indices
    }
    return best_rule, current_best_score, X_sparse_filtered, y_filtered, new_rule_map


def filter_sparse_matrix(X_sparse, y, rule_index, min_pos=5):
    selector = X_sparse[:, rule_index].toarray().flatten() == 1
    selector_negation = ~selector

    X_sparse_filtered_true = X_sparse[selector]
    y_filtered_true = y[selector]

    X_sparse_filtered_false = X_sparse[selector_negation]
    y_filtered_false = y[selector_negation]

    return (X_sparse_filtered_true, y_filtered_true), (
        X_sparse_filtered_false,
        y_filtered_false,
    )


def update_rule_map(X_sparse_filtered, rule_map, min_pos):
    # Remove columns with fewer than min_pos positive entries
    column_sums = X_sparse_filtered.sum(axis=0).A.flatten()
    valid_columns = column_sums >= min_pos
    X_sparse_filtered = X_sparse_filtered[:, valid_columns]

    # Rebuild rule_map with new valid column indices
    old_to_new_indices = {
        old: new for new, old in enumerate(np.where(valid_columns)[0])
    }
    new_rule_map = {
        rule: old_to_new_indices[idx]
        for rule, idx in rule_map.items()
        if idx in old_to_new_indices
    }

    return X_sparse_filtered, new_rule_map


def calculate_false_woe(X_sparse_false, y_false, total_goods, total_bads):
    false_goods = np.sum(y_false)
    return calculate_woe(
        X_sparse_false.sum(), false_goods, total_goods, total_bads
    )  # TODO: fix


def find_best_rule_in_X(
    X_sparse, y, rule_map, total_goods, total_bads, previous_best_score, min_pos=5
):
    best_index, best_rule = find_best_rule(
        X_sparse, y, rule_map, total_goods, total_bads
    )
    current_best_score = calculate_woe(
        X_sparse[:, best_index].sum(),
        y[X_sparse[:, best_index].toarray().flatten() == 1].sum(),
        total_goods,
        total_bads,
    )

    if previous_best_score >= current_best_score:  # Stop if no improvement
        # print("No improvement in WoE score. Stopping refinement.")
        return None, None, None, None, None, None, None, None

    (
        (X_sparse_filtered_true, y_filtered_true),
        (X_sparse_filtered_false, y_filtered_false),
    ) = filter_sparse_matrix(X_sparse, y, best_index)

    X_sparse_filtered_true, new_rule_map_true = update_rule_map(
        X_sparse_filtered_true, rule_map, min_pos
    )
    X_sparse_filtered_false, new_rule_map_false = update_rule_map(
        X_sparse_filtered_false, rule_map, min_pos
    )

    return (
        best_rule,
        current_best_score,
        X_sparse_filtered_true,
        y_filtered_true,
        new_rule_map_true,
        X_sparse_filtered_false,
        y_filtered_false,
        new_rule_map_false,
    )


from graphviz import Digraph


class DecisionTreeNode:
    def __init__(
        self,
        rule=None,
        true_branch=None,
        false_branch=None,
        is_leaf=False,
        score=0.0,
        score_true=None,
        score_false=None,
    ):
        self.rule = rule
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.is_leaf = is_leaf
        self.score = score  # This remains as the base score for any computations.
        self.score_true = score_true  # Score when the rule results in true
        self.score_false = score_false  # Score when the rule results in false

    def __str__(self):
        return f"Rule: {self.rule}, Base Score: {self.score}, Score True: {self.score_true}, Score False: {self.score_false}, Is Leaf: {self.is_leaf}"


def build_decision_tree(
    X_sparse,
    y,
    rule_map,
    previous_best_score,
    total_goods,
    total_bads,
    depth,
    max_depth=3,
    min_samples=10,
    min_pos=5,
    min_improvement=0.00001,
):
    if depth >= max_depth or len(y) < min_samples:
        return DecisionTreeNode(is_leaf=True, score=previous_best_score)

    (
        best_rule,
        current_best_score,
        X_sparse_true,
        y_true,
        rule_map_true,
        X_sparse_false,
        y_false,
        rule_map_false,
    ) = find_best_rule_in_X(
        X_sparse,
        y,
        rule_map,
        total_goods,
        total_bads,
        previous_best_score,
        min_pos=min_pos,
    )

    if (
        best_rule is None
        or (current_best_score - previous_best_score) < min_improvement
    ):
        return DecisionTreeNode(is_leaf=True, score=previous_best_score)

    score_true = (
        current_best_score  # this score is achieved when the rule result is true
    )
    score_false = calculate_false_woe(
        X_sparse_false, y_false, total_goods, total_bads
    )  # calculating score when the rule result is false
    true_branch = build_decision_tree(
        X_sparse_true,
        y_true,
        rule_map_true,
        score_true,
        total_goods,
        total_bads,
        depth + 1,
        max_depth,
        min_samples,
        min_pos,
        min_improvement,
    )
    if not np.isnan(score_false):
        false_branch = build_decision_tree(
            X_sparse_false,
            y_false,
            rule_map_false,
            score_false,
            total_goods,
            total_bads,
            depth + 1,
            max_depth,
            min_samples,
            min_pos,
            min_improvement,
        )
    else:
        false_branch = DecisionTreeNode(is_leaf=True, score=score_true)
    return DecisionTreeNode(
        rule=best_rule,
        true_branch=true_branch,
        false_branch=false_branch,
        score=previous_best_score,
        score_true=score_true,
        score_false=score_false,
    )


from graphviz import Digraph


class DecisionTreePlotter:
    def __init__(self, tree_root):
        self.tree_root = tree_root

    def add_nodes_edges(
        self, dot, node, parent_id=None, parent=None, edge_label=None, depth=0
    ):
        current_id = f"node_{id(node)}_{depth}"
        node_color = "lightgreen" if node.is_leaf else "lightblue"
        if parent_id:
            # Use the parent's score for decisions based on edge label
            if edge_label == "Yes":
                score = (
                    parent.score_true
                    if parent and parent.score_true is not None
                    else None
                )
            else:
                score = (
                    parent.score_false
                    if parent and parent.score_false is not None
                    else None
                )

            # Skip nodes with invalid scores.
            if node.is_leaf and (
                score is None or score <= 0 or np.isinf(score) or np.isnan(score)
            ):
                return

            # Append score to the edge label for visualization, rounded to 2 decimals.
            if score is not None:
                edge_label_formatted = f"{edge_label}\nScore: {round(score, 2)}"
            else:
                edge_label_formatted = edge_label

            dot.edge(
                parent_id,
                current_id,
                label=edge_label_formatted,
                arrowsize="0.6",
                color="black",
            )

        label = (
            f"Leaf\nScore: {round(score, 2)}"
            if node.is_leaf and score is not None
            else node.rule
        )
        dot.node(current_id, label, fillcolor=node_color, style="filled")

        if not node.is_leaf:
            self.add_nodes_edges(
                dot, node.true_branch, current_id, node, "Yes", depth + 1
            )
            self.add_nodes_edges(
                dot, node.false_branch, current_id, node, "No", depth + 1
            )

    def plot_tree(self):
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

        if self.tree_root:
            self.add_nodes_edges(dot, self.tree_root)

        return dot
