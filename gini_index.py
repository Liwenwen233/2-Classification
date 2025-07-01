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
    total = len(dataset)
    if total == 0:         
        return 0.0
    counts = dataset[target_attribute].value_counts()
    impurity = 1.0 - sum((cnt / total) ** 2 for cnt in counts)
    return impurity


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
    total = len(dataset)
    if total == 0:
        return 0.0
    if isinstance(split, set):
        left = dataset[dataset[partition_attribute].isin(split)]
        right = dataset[~dataset[partition_attribute].isin(split)]
    else:
        left = dataset[dataset[partition_attribute] <= split]
        right = dataset[dataset[partition_attribute] > split]
    weighted_impurity = 0.0
    for subset in (left, right):
        weight = len(subset) / total
        if weight > 0:     
            weighted_impurity += weight * calculate_impurity(subset, target_attribute)
    return weighted_impurity


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
    base_impurity = calculate_impurity(dataset, target_attribute)

    impurity_after = calculate_impurity_partitioned(
        dataset,
        target_attribute,
        partition_attribute,
        split,
    )
    return base_impurity - impurity_after