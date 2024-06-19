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
