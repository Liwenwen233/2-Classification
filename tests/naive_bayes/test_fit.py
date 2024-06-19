from naive_bayes import NaiveBayes

#####
# Tests with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the training of the Naive Bayes classifier with the small student dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Fit the Naive Bayes classifier
    naive_bayes.fit(small_student_dataset, "Passed")

    # Check if the target attribute is set correctly
    assert naive_bayes.target_attribute == "Passed"

    # Check if the class labels are set correctly
    assert naive_bayes.class_labels == {"Yes", "No"}

    # Check if the prior probabilities are correct
    assert naive_bayes.prior_probabilities.get_prior_probability("Yes") == 0.5
    assert naive_bayes.prior_probabilities.get_prior_probability("No") == 0.5

    # Check if the likelihoods are correct
    # Attribute 'Age'
    assert (
        naive_bayes.likelihoods.likelihoods["Age"]["Yes"]["mean"] == 24.333333333333332
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Age"]["Yes"]["std"] == 1.5275252316519468
    )
    assert naive_bayes.likelihoods.likelihoods["Age"]["No"]["mean"] == 25.0
    assert naive_bayes.likelihoods.likelihoods["Age"]["No"]["std"] == 1.7320508075688772

    # Attribute 'Major'
    assert (
        naive_bayes.likelihoods.likelihoods["Major"]["CS"]["Yes"] == 0.3333333333333333
    )
    assert naive_bayes.likelihoods.likelihoods["Major"]["CS"]["No"] == 0.0
    assert (
        naive_bayes.likelihoods.likelihoods["Major"]["DS"]["Yes"] == 0.6666666666666666
    )
    assert naive_bayes.likelihoods.likelihoods["Major"]["DS"]["No"] == 1.0

    # Attribute 'Participation'
    assert (
        naive_bayes.likelihoods.likelihoods["Participation"]["High"]["Yes"]
        == 0.6666666666666666
    )
    assert naive_bayes.likelihoods.likelihoods["Participation"]["High"]["No"] == 0.0
    assert naive_bayes.likelihoods.likelihoods["Participation"]["Low"]["Yes"] == 0.0
    assert (
        naive_bayes.likelihoods.likelihoods["Participation"]["Low"]["No"]
        == 0.6666666666666666
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Participation"]["Medium"]["Yes"]
        == 0.3333333333333333
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Participation"]["Medium"]["No"]
        == 0.3333333333333333
    )


#####
# Tests with the small submission dataset
#####


def test_with_small_submission_dataset(small_submission_dataset):
    """
    Test the training of the Naive Bayes classifier with the small submission dataset.
    """

    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Fit the Naive Bayes classifier
    naive_bayes.fit(small_submission_dataset, "Passed")

    # Check if the target attribute is set correctly
    assert naive_bayes.target_attribute == "Passed"

    # Check if the class labels are set correctly
    assert naive_bayes.class_labels == {"Yes", "No"}

    # Check if the prior probabilities are correct
    assert naive_bayes.prior_probabilities.get_prior_probability("Yes") == 0.7
    assert naive_bayes.prior_probabilities.get_prior_probability("No") == 0.3

    # Check if the likelihoods are correct
    # Attribute 'Topic'
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Classification"]["Yes"]
        == 0.2857142857142857
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Classification"]["No"]
        == 0.3333333333333333
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Clustering"]["Yes"]
        == 0.42857142857142855
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Clustering"]["No"]
        == 0.3333333333333333
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Frequent Patterns"]["Yes"]
        == 0.2857142857142857
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Topic"]["Frequent Patterns"]["No"]
        == 0.3333333333333333
    )

    # Attribute 'Knowledge'
    assert (
        naive_bayes.likelihoods.likelihoods["Knowledge"]["High"]["Yes"]
        == 0.2857142857142857
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Knowledge"]["High"]["No"]
        == 0.6666666666666666
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Knowledge"]["Low"]["Yes"]
        == 0.2857142857142857
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Knowledge"]["Low"]["No"]
        == 0.3333333333333333
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Knowledge"]["Medium"]["Yes"]
        == 0.42857142857142855
    )
    assert naive_bayes.likelihoods.likelihoods["Knowledge"]["Medium"]["No"] == 0.0

    # Attribute 'Hours'
    assert (
        naive_bayes.likelihoods.likelihoods["Hours"]["Yes"]["mean"] == 4.428571428571429
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Hours"]["Yes"]["std"] == 1.1338934190276817
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Hours"]["No"]["mean"] == 2.3333333333333335
    )
    assert (
        naive_bayes.likelihoods.likelihoods["Hours"]["No"]["std"] == 1.5275252316519468
    )
