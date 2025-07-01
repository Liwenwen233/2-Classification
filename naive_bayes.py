import pandas as pd
from typing import List, Set
import warnings

from classes.naive_bayes_likelihoods import NaiveBayesLikelihoods
from classes.naive_bayes_prior_probabilities import NaiveBayesPriorProbabilities


class NaiveBayes:
    """
    A Naive Bayes classifier.
    """

    def __init__(self):
        # The target attribute to predict
        self.target_attribute: str = None

        # A full set of possible class labels ( = values of the target attribute)
        self.class_labels: Set[str | int | float] = None

        # The likelihoods of the classifier
        self.likelihoods: NaiveBayesLikelihoods = None

        # The prior probabilities of the classifier
        self.prior_probabilities: NaiveBayesPriorProbabilities = None

    def fit(self, dataset: pd.DataFrame, target_attribute: str):
        """
        Fit the Naive Bayes classifier to the training dataset.
        Sets the target attribute and the class labels.
        Calculates the prior probabilities, and the likelihoods.

        Parameters:
        dataset (pd.DataFrame): The training dataset
        target_attribute (str): The target attribute to predict
        """
        # Make sure that the target_attribute is in the dataset
        if target_attribute not in dataset.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not in dataset.")

        # TODO
        self.target_attribute = target_attribute
        self.class_labels = set(dataset[target_attribute].dropna().unique())
        if not self.class_labels:
            raise ValueError("No class labels found in the dataset.")
        self.prior_probabilities = self._calculate_prior_probabilities(dataset)
        self.likelihoods = self._calculate_likelihoods(dataset)
        return self

    def _calculate_prior_probabilities(
        self, dataset: pd.DataFrame
    ) -> NaiveBayesPriorProbabilities:
        """
        Calculate the prior probability for each class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesPriorProbabilities: The prior probabilities for each class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO
        total = len(dataset)
        if total == 0:
            raise ValueError("Cannot calculate priors on empty dataset.")
        counts = dataset[self.target_attribute].value_counts()
        priors = NaiveBayesPriorProbabilities()
        for class_label, count in counts.items():
            prob = count / total
            priors.add_prior_probability(class_label, prob)
        return priors

    def _calculate_likelihoods(self, dataset: pd.DataFrame) -> NaiveBayesLikelihoods:
        """
        Calculate the likelihoods for each attribute and class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesLikelihoods: The likelihoods for each attribute and class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO
        if self.class_labels is None:
            self.class_labels = set(dataset[self.target_attribute].dropna().unique())
            if not self.class_labels:
                raise ValueError("No class labels found in the dataset.")
        likelihoods = NaiveBayesLikelihoods()
        for attribute in dataset.columns:
            if attribute == self.target_attribute:
                continue
            col = dataset[attribute]
            is_numeric = pd.api.types.is_numeric_dtype(col)
            if not is_numeric:
                global_values = col.dropna().unique()
            for cls in self.class_labels:
                subset = dataset[dataset[self.target_attribute] == cls]
                n_cls = len(subset)
                if n_cls == 0:
                    warnings.warn(
                        f"No examples of class '{cls}' to compute likelihood for '{attribute}'."
                    )
                    continue
                if is_numeric:
                    vals = subset[attribute].dropna()
                    mu = vals.mean()
                    sigma = vals.std(ddof=1)
                    if sigma == 0:
                        sigma = 1e-6
                    likelihoods.add_continuous_likelihood(attribute, cls, mu, sigma)
                else:
                    for val in global_values:
                        cnt = int((subset[attribute] == val).sum())
                        prob = cnt / n_cls
                        likelihoods.add_categorical_likelihood(attribute, val, cls, prob)
        return likelihoods

    def predict(self, dataset: pd.DataFrame) -> List[str | int | float]:
        """
        Predict the target attribute for a given dataset.

        Parameters:
        dataset (pd.DataFrame): The dataset to predict the target attribute for

        Returns:
        List[str | int | float]: A list of predicted class labels
        """

        # If the likelihoods or/and the prior probabilities are not calculated yet, raise an error
        if self.likelihoods is None or self.prior_probabilities is None:
            raise ValueError("Model not trained yet.")

        # TODO
        predictions: List[str | int | float] = []
        for _, row in dataset.iterrows():
            pred = self._predict_tuple(row)
            predictions.append(pred)

        return predictions

    def _predict_tuple(self, tuple: pd.Series) -> str | int | float:
        """
        Predict the target attribute for a given row in the dataset.

        Parameters:
        tuple (pd.Series): The row in the dataset to predict the target attribute for

        Returns:
        str | int | float: The predicted class label
        """
        # TODO
        if (
            self.target_attribute is None
            or self.prior_probabilities is None
            or self.likelihoods is None
        ):
            raise ValueError("Classifier not fitted. Call fit() first.")

        best_label = None
        best_score = -float("inf")
        for cls in self.class_labels:
            score = self.prior_probabilities.get_prior_probability(cls)
            for attr, val in tuple.items():
                if attr == self.target_attribute:
                    continue
                lik = self.likelihoods.get_likelihood(attr, val, cls)
                score *= lik
            if best_label is None or score > best_score:
                best_label = cls
                best_score = score
        return best_label
