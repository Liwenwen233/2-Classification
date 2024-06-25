import pandas as pd
from math import log
from typing import List, Set

"""
Collection of functions to calculate the impurity and the gini index of attributes in a dataset.
"""


def calculate_impurity(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated impurity
    """
    # TODO


def calculate_impurity_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).
    """
    # TODO


def calculate_gini_index(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the Gini index (= reduction of impurity) for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the Gini index for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).

    Returns:
    float: The calculated Gini index
    """
    # TODO
