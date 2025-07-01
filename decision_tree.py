import itertools
import pandas as pd
from typing import List, Tuple, Set

import gini_index
import information_gain
from classes.decision_tree_node import DecisionTreeNode
from classes.decision_tree_leaf_node import DecisionTreeLeafNode
from classes.decision_tree_internal_node import DecisionTreeInternalNode
from classes.decision_tree_branch import DecisionTreeBranch
from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome
from classes.decision_tree_decision_outcome_above import (
    DecisionTreeDecisionOutcomeAbove,
)
from classes.decision_tree_decision_outcome_below_equal import (
    DecisionTreeDecisionOutcomeBelowEqual,
)
from classes.decision_tree_decision_outcome_equals import (
    DecisionTreeDecisionOutcomeEquals,
)
from classes.decision_tree_decision_outcome_in_list import (
    DecisionTreeDecisionOutcomeInList,
)


class DecisionTree:
    """
    A Decision Tree classifier.
    """

    def __init__(self):
        """
        Initialize the DecisionTree object.
        """

        # Function fit will later populate this variable
        self.target_attribute: str = None

        # Function fit will later produce a decision tree
        self.tree: DecisionTreeNode = None

    def fit(
        self,
        dataset: pd.DataFrame,
        target_attribute: str,
        attribute_selection_method: str,
    ):
        """
        Fit decision tree on a given dataset and target attribute, using a specified attribute selection method.

        Parameters:
        dataset (pd.DataFrame): The dataset to fit the decision tree on
        target_attribute (str): The target attribute to predict
        attribute_selection_method (str): The attribute selection method to use
        """
        # Make sure that the target_attribute is in the dataset
        if target_attribute not in dataset.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not in dataset.")

        # Make sure that the attribute_selection_method is valid
        if attribute_selection_method not in [
            "information_gain",
            "gini_index",
        ]:
            raise ValueError(
                f"Attribute selection method '{attribute_selection_method}' not valid (select either 'information_gain' or 'gini_index')."
            )

        # TODO
        self.target_attribute = target_attribute
        attributes = [col for col in dataset.columns if col != target_attribute]
        self.tree = self._build_tree(dataset, attributes, attribute_selection_method)
        return self

    def _build_tree(
        self,
        data: pd.DataFrame,
        attribute_list: List[str],
        attribute_selection_method: str,
    ) -> DecisionTreeNode:
        """
        Recursively build the decision tree.

        Parameters:
        data (pd.DataFrame): The (partial) dataset to build the decision tree with
        attribute_list (List[str]): The list of attributes to consider
        attribute_selection_method (str): The attribute selection method to use

        Returns:
        DecisionTreeNode: The root node of the decision tree
        """
        # TODO
        target_vals = data[self.target_attribute]
        if target_vals.nunique() == 1:
            return DecisionTreeLeafNode(target_vals.iloc[0])
        if not attribute_list:
            majority = target_vals.mode()[0]
            return DecisionTreeLeafNode(majority)
        best_attr, outcomes = self._find_best_split(
            data, attribute_list, attribute_selection_method
        )
        if not outcomes:
            majority = target_vals.mode()[0]
            return DecisionTreeLeafNode(majority)
        branches = []
        majority = target_vals.mode()[0]
        for outcome in outcomes:
            mask = data[best_attr].apply(lambda v: outcome.value_matches(v))
            subset = data[mask]
            if subset.empty:
                child = DecisionTreeLeafNode(majority)
            else:
                remaining_attrs = [a for a in attribute_list if a != best_attr]
                child = self._build_tree(
                    subset, remaining_attrs, attribute_selection_method
                )
            branches.append(DecisionTreeBranch(outcome, child))
        return DecisionTreeInternalNode(best_attr, branches)
    
    def _find_best_split(
        self,
        data: pd.DataFrame,
        attribute_list: List[str],
        attribute_selection_method: str,
    ) -> Tuple[str, List[DecisionTreeDecisionOutcome]]:
        """
        Find the best split for a given dataset and attribute list. Finding the best split includes finding the best attribute to split on and also (depending on the attribute selection method) the best set of outcomes to split on this attribute.

        Parameters:
        data (pd.DataFrame): The dataset to find the best splitting attribute for
        attribute_list (List[str]): The list of attributes to consider
        attribute_selection_method (str): The attribute selection method to use

        Returns:
        str: The attribute to split on
        List[DecisionTreeDecisionOutcome]: The outcomes a split on this attribute should have
        """
        # TODO
        if not attribute_list:
            raise ValueError("No attributes available for splitting.")
        best_attr: str = None
        best_outcomes: List[DecisionTreeDecisionOutcome] = []
        best_score: float = -float("inf")
        for attr in attribute_list:
            if attribute_selection_method == "information_gain":
                score, outcomes = self._calculate_information_gain(data, attr)
            elif attribute_selection_method == "gini_index":
                score, outcomes = self._calculate_gini_index(data, attr)
            else:
                raise ValueError(f"Unknown attribute_selection_method: {attribute_selection_method!r}")
            if score > best_score:
                best_score = score
                best_attr = attr
                best_outcomes = outcomes
        return best_attr, best_outcomes

    def _calculate_information_gain(
        self, data: pd.DataFrame, attribute: str
    ) -> Tuple[float, List[DecisionTreeDecisionOutcome]]:
        """
        Calculate the (best) information gain for a given attribute in a dataset.

        Parameters:
        data (pd.DataFrame): The dataset to calculate the information gain for
        attribute (str): The attribute to calculate the information gain for

        Returns:
        float: The calculated information gain
        List[DecisionTreeDecisionOutcome]: The outcomes the best split of this attribute has
        """
        # If self.target_attribute is not set, raise an error
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # If the attribute is not in the dataset, raise an error
        if attribute not in data.columns:
            raise ValueError(f"Attribute '{attribute}' not in dataset.")

        # TODO
        col = data[attribute]
        is_numeric = pd.api.types.is_numeric_dtype(col)
        if is_numeric:
            unique_vals = sorted(col.dropna().unique())
            if len(unique_vals) <= 1:
                return 0.0, []
            best_gain = -float("inf")
            best_threshold = None
            for i in range(len(unique_vals) - 1):
                threshold = (unique_vals[i] + unique_vals[i + 1]) / 2
                gain = information_gain.calculate_information_gain(
                    data,
                    self.target_attribute,
                    attribute,
                    split_value=threshold,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
            if best_threshold is None or best_gain <= 0:
                return 0.0, []
            outcomes = [
                DecisionTreeDecisionOutcomeBelowEqual(best_threshold),
                DecisionTreeDecisionOutcomeAbove(best_threshold),
            ]
            return best_gain, outcomes
        else:
            gain = information_gain.calculate_information_gain(
                data,
                self.target_attribute,
                attribute,
                split_value=None,        
            )
            categories = col.dropna().unique()
            if len(categories) == 0 or gain <= 0:
                return 0.0, []
            outcomes = [
                DecisionTreeDecisionOutcomeEquals(cat) for cat in categories
            ]
            return gain, outcomes

    def _calculate_gini_index(
        self, data: pd.DataFrame, attribute: str
    ) -> Tuple[float, List[DecisionTreeDecisionOutcome]]:
        """
        Calculate the (best) gini index for a given attribute in a dataset.

        Parameters:
        data (pd.DataFrame): The dataset to calculate the gini index for
        attribute (str): The attribute to calculate the gini index for

        Returns:
        float: The calculated gini index (reduction of impurity)
        List[DecisionTreeDecisionOutcome]: The outcomes the best split of this attribute has
        """
        # If self.target_attribute is not set, raise an error
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # If the attribute is not in the dataset, raise an error
        if attribute not in data.columns:
            raise ValueError(f"Attribute '{attribute}' not in dataset.")

        # TODO
        col = data[attribute]
        is_numeric = pd.api.types.is_numeric_dtype(col)
        if is_numeric:
            unique_vals = sorted(col.dropna().unique())
            if len(unique_vals) <= 1:
                return 0.0, []
            best_gain = -float("inf")
            best_threshold = None
            for i in range(len(unique_vals) - 1):
                thresh = (unique_vals[i] + unique_vals[i + 1]) / 2
                gain = gini_index.calculate_gini_index(
                    data,
                    self.target_attribute,
                    attribute,
                    split=thresh,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = thresh
            if best_threshold is None or best_gain <= 0:
                return 0.0, []
            outcomes = [
                DecisionTreeDecisionOutcomeBelowEqual(best_threshold),
                DecisionTreeDecisionOutcomeAbove(best_threshold),
            ]
            return best_gain, outcomes
        else:
            unique_cats = list(col.dropna().unique())
            m = len(unique_cats)
            if m <= 1:
                return 0.0, []
            base_impurity = gini_index.calculate_impurity(data, self.target_attribute)
            best_gain = -float("inf")
            best_subset: Set = set()
            from itertools import combinations
            for r in range(1, m // 2 + 1):
                for subset in combinations(unique_cats, r):
                    subset_set = set(subset)
                    imp_after = gini_index.calculate_impurity_partitioned(
                        data,
                        self.target_attribute,
                        attribute,
                        split=subset_set,
                    )
                    gain = base_impurity - imp_after
                    if gain > best_gain:
                        best_gain = gain
                        best_subset = subset_set
            if best_gain <= 0:
                return 0.0, []
            other = set(unique_cats) - best_subset
            outcomes = [
                DecisionTreeDecisionOutcomeInList(list(best_subset)),
                DecisionTreeDecisionOutcomeInList(list(other)),
            ]
            return best_gain, outcomes

    
    def predict(self, dataset: pd.DataFrame) -> List[str | int | float]:
        """
        Predict the target attribute for a given dataset.

        Parameters:
        dataset (pd.DataFrame): The dataset to predict the target attribute for

        Returns:
        List[str | int | float]: A list of predicted class labels
        """

        # If the tree is not fitted, raise an error
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit method first.")

        # TODO
        predictions: List[str | int | float] = []
        for _, row in dataset.iterrows():
            node = self.tree
            while isinstance(node, DecisionTreeNode) and not isinstance(node, DecisionTreeLeafNode):
                branches = node.get_branches()
                chosen_child = None
                for branch in branches:
                    outcome = next(
                        (v for v in vars(branch).values() if isinstance(v, DecisionTreeDecisionOutcome)),
                        None,
                    )
                    child = next(
                        (v for v in vars(branch).values() if isinstance(v, DecisionTreeNode)),
                        None,
                    )
                    if outcome is None or child is None:
                        continue
                    if outcome.value_matches(row[node.get_label()]):
                        chosen_child = child
                        break
                if chosen_child is None:
                    break
                node = chosen_child
            if isinstance(node, DecisionTreeLeafNode):
                predictions.append(node.get_label())
            else:
                majority = dataset[self.target_attribute].mode()[0]
                predictions.append(majority)
        return predictions

    def _predict_tuple(
        self, tuple: pd.Series, node: DecisionTreeNode
    ) -> str | int | float:
        """
        Predict the target attribute for a given row in the dataset.
        This is a recursive function that traverses the decision tree until a leaf node is reached.

        Parameters:
        tuple (pd.Series): The row to predict the target attribute for
        node (DecisionTreeNode): The current node in the decision tree

        Returns:
        str | int | float: The predicted class label
        """
        # TODO
        if isinstance(node, DecisionTreeLeafNode):
            return node.get_label()
        assert isinstance(node, DecisionTreeInternalNode), "Node must be internal or leaf"
        attr = node.get_label()
        val = tuple[attr]
        for branch in node.get_branches():
            if branch.value_matches(val):
                return self._predict_tuple(tuple, branch.get_branch_node())
        raise ValueError(f"No branch for value {val!r} of attribute '{attr}'")
