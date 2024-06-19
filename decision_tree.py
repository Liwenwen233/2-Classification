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
