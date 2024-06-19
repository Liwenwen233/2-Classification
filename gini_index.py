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
    # If the split is a set, the partition attribute is discrete-valued
    if isinstance(split, set):
        # Compute the two partitions
        mask = dataset[partition_attribute].apply(lambda x: x in split)
        partition = dataset[mask]
        partition2 = dataset[~mask]

        # Calculate the impurity for each partition and return the weighted sum
        impurity = partition.shape[0] / dataset.shape[0] * calculate_impurity(
            partition, target_attribute
        ) + partition2.shape[0] / dataset.shape[0] * calculate_impurity(
            partition2, target_attribute
        )

        return impurity
    # If the split is a single value, the partition attribute is continuous-valued
    else:
        # Compute the two partitions
        partition = dataset[dataset[partition_attribute] <= split]
        partition2 = dataset[dataset[partition_attribute] > split]

        # Calculate the impurity for each partition and return the weighted sum
        impurity = partition.shape[0] / dataset.shape[0] * calculate_impurity(
            partition, target_attribute
        ) + partition2.shape[0] / dataset.shape[0] * calculate_impurity(
            partition2, target_attribute
        )

        return impurity


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
    # Calculate the impurity of the dataset
    impurity = calculate_impurity(dataset, target_attribute)

    # Calculate the impurity of the partitioned dataset
    impurity_partitioned = calculate_impurity_partitioned(
        dataset, target_attribute, partition_attribute, split
    )

    # Calculate the Gini index
    return impurity - impurity_partitioned
