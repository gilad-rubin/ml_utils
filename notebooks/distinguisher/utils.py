import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

#TODO: deprecate
def sort_categories(X, y, categorical_columns, scoring_function):
    categories_sorted = {}
    for col in categorical_columns:
        category_scores = scoring_function(X[col], y)
        sorted_categories = category_scores.sort_values().index.tolist()
        categories_sorted[col] = sorted_categories
    return categories_sorted


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
    total_bads = len(y) - total_goods
    grouped = y.groupby(x).agg(["sum", "count"])
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

    def fit(self, X, y):
        for col in X.columns:
            if X[col].isna().any():
                # Handle NaN values by including them in the scoring process if necessary
                # Here we assume NaNs are handled externally in the scoring function if needed
                self.sorted_categories[col] = [np.nan]
            else:
                self.sorted_categories[col] = []
                
            category_scores = self.scoring_function(X[col].dropna(), y[X[col].notna()])
            sorted_categories = category_scores.sort_values().index.tolist()
            # Include NaN handling by adding NaN at the end with index -1
            self.sorted_categories[col] = sorted_categories
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            if col in self.sorted_categories:
                category_mapping = {category: idx for idx, category in enumerate(self.sorted_categories[col])}
                # Assign -1 to NaN values
                category_mapping[np.nan] = -1
                X_transformed[col] = X_transformed[col].map(category_mapping).fillna(-1)
        return X_transformed

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
                    category_mapping = {category: idx for idx, category in enumerate(sorted_categories) if category is not np.nan}
                    category_mapping[np.nan] = self.encoded_missing_value  # Explicit NaN mapping
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

    def inverse_transform(self, encoded_values):
        decoded_frames = {}
        for col in encoded_values.columns:
            if col in self.category_mappings:
                # We need to inverse map while considering the handling of missing values
                inverse_map = {v: k for k, v in self.category_mappings[col].items() if v != -1}
                inverse_map[self.encoded_missing_value] = np.nan  # Map encoded_missing_value back to NaN
                decoded_frames[col] = encoded_values[col].map(inverse_map)
        return pd.DataFrame(decoded_frames)

# TODO: deprecate
# TODO: make it get the order of categories
# TODO: think of taking "convert_to_categorical" out of the class
# TODO: perhaps remove this class altogether

class Encoder:
    def __init__(self):
        self.encoder = OrdinalEncoder()
        self.category_mappings = {}

    def fit_transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include=["category", "object"]).columns:
            original_categories = list(X[col].cat.categories)
            X[col] = self.encoder.fit_transform(X[[col]])[:, 0]
            self.category_mappings[col] = original_categories
        return X
    
    #TODO: Take out!
    def convert_to_categorical(self, feature, value, is_lower_bound):
        categories = self.category_mappings.get(feature, [])
        index = int(value)  # Assuming value is already the correct integer index
        if index >= len(
            categories
        ):  # Ensure the index does not exceed the last category
            index = len(categories) - 1
        if index < 0:  # Ensure the index is not negative
            index = 0

        if is_lower_bound:
            res = categories[index + 1 :]
        else:
            res = categories[: index + 1]

        return res


class BinningStrategy:
    def calculate_bins(self, data, max_bins):
        raise NotImplementedError("Each strategy must implement the calculate_bins method.")
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
            if data.dtype in ['object', 'category'] or pd.api.types.is_string_dtype(data):
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

# TODO: deprecate
# TODO: refactor + think of making this more general (e.g. allowing for different algorithms)
# TODO: add error handling - IQR == 0; add better rules for "else"
# TODO: handle NaN values?
def calculate_optimal_bins(X, max_bins=20, min_bins=2, factor=1.5):
    n_bins = []
    for column in X.columns:
        if len(pd.unique(X[column])) < max_bins:
            # For categorical data, use a default number of bins
            n_bins_col = min(len(pd.unique(X[column])) - 1, max_bins)
            n_bins.append(max(min_bins, n_bins_col))
        else:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            n = len(X[column].dropna())  # Exclude NaN values for the calculation

            # Calculate bin width using Freedman-Diaconis rule
            bin_width = factor * IQR / (n ** (1 / 3))

            # Calculate number of bins
            if bin_width > 0:
                optimal_bins = int((X[column].max() - X[column].min()) / bin_width)
                optimal_bins = max(min(optimal_bins, max_bins), min_bins)
            else:
                optimal_bins = min_bins

            n_bins.append(optimal_bins)

    return n_bins

#TODO: depracate
# TODO: make it get optimal bins for each column
# TODO: add .fit and .transform
class PandasQCutDiscretizer:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.left_edges = {}
        self.right_edges = {}

    def fit_transform(self, X):
        X = X.copy()
        for col in X.columns:
            if X[col].dtype in [
                np.float64,
                np.int64,
            ]:  # We only discretize numeric columns
                X[col] = pd.qcut(X[col], self.n_bins, duplicates="drop")
                self.left_edges[col] = [
                    bin.left for bin in X[col].unique() if bin is not np.nan
                ]
                self.right_edges[col] = [
                    bin.right for bin in X[col].unique() if bin is not np.nan
                ]
        return X

    def get_bin_edges(self):
        return self.left_edges, self.right_edges


# TODO: where to put this? (in a class or outside)
# TODO: consider changing the logic. instead of giving it a path - just change the last rule and add both options to the applied path
def negate_last_rule(path):
    if path.rules:
        new_rules = path.rules[:-1] + [path.rules[-1].negate_rule()]
        return Path(new_rules)
    return path


# TODO: decide on naming - Node & Path. Rule & Path?
# TODO: fill score attribute (?) perhaps link to BinaryRuleScore?
# TODO: where to put "negate_rule"?
# TODO: __str__ vs __repr__ vs get_path_rule
class Rule:
    def __init__(self, feature, operator, value, score=0):
        self.feature = feature
        self.operator = operator
        self.value = value
        self.score = score

    def __str__(self):
        if self.operator in ["isna", "notnull"]:
            return f"`{self.feature}`.{self.operator}()"
        else:
            return f"`{self.feature}` {self.operator} {self.value}"

    # TODO: improve the get_mask ifs
    def get_mask(self, X):
        if self.operator in ["isna", "notnull"]:
            if self.operator == "isna":
                return X[self.feature].isna()
            else:
                return X[self.feature].notnull()
        else:
            return eval(f"X['{self.feature}'] {self.operator} {self.value}")

    def negate_rule(self):
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
        new_operator = negation_map[self.operator]
        return Rule(self.feature, new_operator, self.value)


# TODO: what is the relationship between this class and discretizer?
class RuleGenerator:
    def __init__(self, discretizer):
        self.discretizer = discretizer

    def generate_all_rules(self, X):
        rules = []
        for feature in X.columns:
            left_edges = self.discretizer.left_edges[feature]
            right_edges = self.discretizer.right_edges[feature]
            rules += [Rule(feature, ">", edge) for edge in left_edges]
            rules += [Rule(feature, "<=", edge) for edge in right_edges]

            if X[feature].isna().any():
                rules.append(Rule(feature, "isna", None))

        return rules


# TODO: get_path_rule - where to use "()"? here or in rule?
# TODO: what's the relationship between __repr__ and __str__ and get_path_rule?
class Path:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else []

    def get_mask(self, X):
        mask = np.ones(len(X), dtype=bool)
        for rule in self.rules:
            mask &= rule.get_mask(X)
        return mask

    def get_path_rule(self):
        return " and ".join([str(rule) for rule in self.rules])

    def __repr__(self):
        return "Path: " + " and ".join([str(rule) for rule in self.rules])


# TODO add positive and negative counts here and in the rest code
# TODO define order of scoring - WoE, Recall, Precision (used for sorting and filtering)
# TODO: design a general RuleScore class. decide on this class's interaction with other related classes
class BinaryRuleScore:
    def __init__(self, recall, precision, WoE):
        self.recall = recall
        self.precision = precision
        self.WoE = WoE

    def __repr__(self):
        return f"BinaryRuleScore (Recall: {self.recall}, Precision: {self.precision}, WoE: {self.WoE})"


# TODO: extract WoE from here and put it in a separate class
# TODO: just go over the first scores and see if there are ties. if so - take the ties and go over the next score
# TODO: handle all the "== 0" in a better way
# TODO: quickly remove rules that have less than min_samples (min_recall) or are trivial - cover the whole dataset?
class BinaryRuleEvaluator:
    def __init__(self):
        pass

    def evaluate(self, rule, X, y):
        mask = rule.get_mask(X)

        true_positives = y[mask].sum()
        total_positives = y.sum()
        predicted_positives = mask.sum()
        total_cases = len(y)

        recall = true_positives / total_positives if total_positives != 0 else 0
        precision = (
            true_positives / predicted_positives if predicted_positives != 0 else 0
        )

        total_negatives = total_cases - total_positives
        false_positives = predicted_positives - true_positives

        if true_positives == 0:
            return BinaryRuleScore(recall, precision, float("-inf"))

        if false_positives == 0:
            false_positives = 1  # Avoid division by zero

        if false_positives and total_negatives:
            WoE = np.log(
                (true_positives / total_positives) / (false_positives / total_negatives)
            )
        else:
            WoE = float("inf")

        return BinaryRuleScore(recall, precision, WoE)


# TODO: consider adding this to the BinaryRuleEvaluator class
class BinaryRuleFilter:
    def __init__(self, min_recall, min_precision, min_WoE):
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.min_WoE = min_WoE

    def apply(self, rule_score):
        return (
            rule_score.recall >= self.min_recall
            and rule_score.precision >= self.min_precision
            and rule_score.WoE >= self.min_WoE
        )


# TODO: make the default decisions for Distinguisher()
# TODO: support multi-class
# TODO: support regression
# TODO: rethink classes and calls
class Distinguisher:
    def __init__(self, encoder, discretizer, rule_generator, evaluator):
        self.encoder = encoder
        self.discretizer = discretizer
        self.rule_generator = rule_generator
        self.evaluator = evaluator
        self.rules = []

    # TODO: add max_depth
    def find_rules(self, X, y, rule_filter):
        # TODO: perhaps add preprocess function?
        informative_cols = [col for col in X.columns if len(pd.unique(X[col])) > 1]
        X_encoded = self.encoder.fit_transform(X[informative_cols])
        X_preprocessed = self.discretizer.fit_transform(X_encoded)

        # TODO: consider the use of self... here and in the rest of the function
        self.all_rules = self.rule_generator.generate_all_rules(X_preprocessed)

        final_rules = self.find_rules_recursive(
            X_encoded, y, chosen_rules=[], applied_path=None, rule_filter=rule_filter
        )

        self.beautify_rules(final_rules, self.encoder, X)
        return final_rules

    # TODO: rename ?
    # TODO: add self.max_depth? or max_depth here?
    # TODO: enable the building of multiple nodes under the root where it's implicit that it's relevant to the negative of all the rules that have been applied so far
    def find_rules_recursive(
        self, X, y, chosen_rules, applied_path, rule_filter, best_score=float("-inf")
    ):
        # print(f"Evaluating with applied path: {applied_path}")
        # TODO: improve logic, refactor
        applied_rules = [] if applied_path is None else applied_path.rules
        evaluated_rules = []
        for rule in self.all_rules:
            rule_combination = Path(applied_rules + [rule])
            rule_score = self.evaluator.evaluate(rule_combination, X, y)
            # print(rule_combination, rule_score)
            evaluated_rules.append((rule_combination, rule_score))

        # Filter rules based on evaluation
        filtered_rules = [
            rule for rule in evaluated_rules if rule_filter.apply(rule[1])
        ]
        if not filtered_rules:
            print("No valid rules after filtering.")
            return chosen_rules

        # TODO: negate negative rules (with max abs score). beware of -inf.
        best_rule = max(
            filtered_rules, key=lambda x: x[1].WoE
        )  # Use WoE as the primary score metric
        current_best_score = best_rule[1].WoE
        best_rule = best_rule[0]

        # convert ">" to "is_better" - handles <, > and multiple sorting
        # refactor. there are repeptitions
        if current_best_score > best_score:
            # TODO: understand why there are duplicates here and cancel them
            if best_rule not in chosen_rules:
                chosen_rules.append(best_rule)
                best_rule.score = current_best_score  # TODO: fix this. it's a path not a rule... confusing. also not showing in plot
            self.find_rules_recursive(
                X, y, chosen_rules, best_rule, rule_filter, current_best_score
            )

            neg_rule = negate_last_rule(best_rule)
            neg_score = self.evaluator.evaluate(neg_rule, X, y)
            if rule_filter.apply(neg_score):
                if neg_rule not in chosen_rules:
                    chosen_rules.append(neg_rule)
                    neg_rule.score = neg_score.WoE
                self.find_rules_recursive(
                    X, y, chosen_rules, neg_rule, rule_filter, neg_score.WoE
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
