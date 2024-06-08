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
    
    
    #TODO: deprecate
def sort_categories(X, y, categorical_columns, scoring_function):
    categories_sorted = {}
    for col in categorical_columns:
        category_scores = scoring_function(X[col], y)
        sorted_categories = category_scores.sort_values().index.tolist()
        categories_sorted[col] = sorted_categories
    return categories_sorted

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