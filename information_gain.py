import pandas as pd
from math import log
from typing import List

"""
Collection of functions to calculate the entropy, information and 
information gain of attributes in a dataset.
"""


def calculate_entropy(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the entropy for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the entropy for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated entropy (= expected information)
    """
    # TODO
    value_counts = dataset[target_attribute].value_counts()
    total = len(dataset)
    entropy = 0.0
    for count in value_counts:
        p_i = count / total
        if p_i > 0:                   
            entropy -= p_i * log(p_i, 2)
    return entropy


def calculate_information_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information for a discrete-valued partition attribute. If set to a value, the function will calculate the information for a continuous-valued partition attribute.
    """
    # TODO
    total = len(dataset)
    if total == 0:      
        return 0.0
    if split_value is None:
        partitions = (
            dataset.groupby(partition_attribute, dropna=False)
            if partition_attribute in dataset.columns
            else []
        )
        info = 0.0
        for _, subset in partitions:
            weight = len(subset) / total
            if weight > 0:
                info += weight * calculate_entropy(subset, target_attribute)
        return info
    else:
        below_equal = dataset[dataset[partition_attribute] <= split_value]
        above = dataset[dataset[partition_attribute] > split_value]
        info = 0.0
        for subset in (below_equal, above):
            weight = len(subset) / total
            if weight > 0:
                info += weight * calculate_entropy(subset, target_attribute)
        return info

def calculate_information_gain(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information gain for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information gain for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information gain for a discrete-valued partition attribute. If set to a value, the function will calculate the information gain for a continuous-valued partition attribute.

    Returns:
    float: The calculated information gain
    """
    # TODO
    base_entropy = calculate_entropy(dataset, target_attribute)
    info_after_split = calculate_information_partitioned(
        dataset,
        target_attribute,
        partition_attribute,
        split_value,
    )
    information_gain = base_entropy - info_after_split
    return information_gain
