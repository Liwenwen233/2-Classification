from naive_bayes import NaiveBayes

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the calculation of the likelihoods with the small student dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the likelihoods
    likelihoods = naive_bayes._calculate_likelihoods(small_student_dataset)

    # Check if the likelihoods are correct
    # Attribute 'Age'
    assert likelihoods.likelihoods["Age"]["Yes"]["mean"] == 24.333333333333332
    assert likelihoods.likelihoods["Age"]["Yes"]["std"] == 1.5275252316519468
    assert likelihoods.likelihoods["Age"]["No"]["mean"] == 25.0
    assert likelihoods.likelihoods["Age"]["No"]["std"] == 1.7320508075688772

    # Attribute 'Major'
    assert likelihoods.likelihoods["Major"]["CS"]["Yes"] == 0.3333333333333333
    assert likelihoods.likelihoods["Major"]["CS"]["No"] == 0.0
    assert likelihoods.likelihoods["Major"]["DS"]["Yes"] == 0.6666666666666666
    assert likelihoods.likelihoods["Major"]["DS"]["No"] == 1.0

    # Attribute 'Participation'
    assert likelihoods.likelihoods["Participation"]["High"]["Yes"] == 0.6666666666666666
    assert likelihoods.likelihoods["Participation"]["High"]["No"] == 0.0
    assert likelihoods.likelihoods["Participation"]["Low"]["Yes"] == 0.0
    assert likelihoods.likelihoods["Participation"]["Low"]["No"] == 0.6666666666666666
    assert (
        likelihoods.likelihoods["Participation"]["Medium"]["Yes"] == 0.3333333333333333
    )
    assert (
        likelihoods.likelihoods["Participation"]["Medium"]["No"] == 0.3333333333333333
    )


#####
# Test with the small submission dataset
#####


def test_with_small_submission_dataset(small_submission_dataset):
    """
    Test the calculation of the likelihoods with the small submission dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the likelihoods
    likelihoods = naive_bayes._calculate_likelihoods(small_submission_dataset)

    # Check if the likelihoods are correct
    # Attribute "Topic"
    assert (
        likelihoods.likelihoods["Topic"]["Classification"]["Yes"] == 0.2857142857142857
    )
    assert (
        likelihoods.likelihoods["Topic"]["Classification"]["No"] == 0.3333333333333333
    )
    assert likelihoods.likelihoods["Topic"]["Clustering"]["Yes"] == 0.42857142857142855
    assert likelihoods.likelihoods["Topic"]["Clustering"]["No"] == 0.3333333333333333
    assert (
        likelihoods.likelihoods["Topic"]["Frequent Patterns"]["Yes"]
        == 0.2857142857142857
    )
    assert (
        likelihoods.likelihoods["Topic"]["Frequent Patterns"]["No"]
        == 0.3333333333333333
    )

    # Attribute "Knowledge"
    assert likelihoods.likelihoods["Knowledge"]["High"]["Yes"] == 0.2857142857142857
    assert likelihoods.likelihoods["Knowledge"]["High"]["No"] == 0.6666666666666666
    assert likelihoods.likelihoods["Knowledge"]["Low"]["Yes"] == 0.2857142857142857
    assert likelihoods.likelihoods["Knowledge"]["Low"]["No"] == 0.3333333333333333
    assert likelihoods.likelihoods["Knowledge"]["Medium"]["Yes"] == 0.42857142857142855
    assert likelihoods.likelihoods["Knowledge"]["Medium"]["No"] == 0.0

    # Attribute "Hours"
    assert likelihoods.likelihoods["Hours"]["Yes"]["mean"] == 4.428571428571429
    assert likelihoods.likelihoods["Hours"]["Yes"]["std"] == 1.1338934190276817
    assert likelihoods.likelihoods["Hours"]["No"]["mean"] == 2.3333333333333335
    assert likelihoods.likelihoods["Hours"]["No"]["std"] == 1.5275252316519468
