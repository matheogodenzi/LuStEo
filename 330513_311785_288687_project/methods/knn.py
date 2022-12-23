import numpy as np
from utils import normalize_fn

class KNN(object):
    """
        kNN classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)


    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        # first checks if "k" was passed as a kwarg.
        if "k" in kwargs:
            self.k = kwargs["k"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) > 0 :
            self.k = args[0]
        # if there were no args or kwargs passed, we set the k to 3 (default value).
        else:
            self.k = 3

    def euclidean_dist(self, example, training_examples):
        return np.sqrt(np.sum((training_examples - example)**2, 1))

    def find_k_nearest_neighbors(self, k, distances):
        indices = np.argsort(distances)
        return indices[:k]

    def predict_label(self, neighbor_labels):
        return np.argmax(np.bincount(neighbor_labels))

    def kNN_one_example(self, unlabeled_example, training_features, training_labels, k):
        # Normalize the data
        mean_val = training_features.mean(0,keepdims=True)
        std_val = training_features.std(0,keepdims=True)
        norm_train = normalize_fn(training_features, mean_val,std_val)
        norm_test_single = normalize_fn(unlabeled_example, mean_val, std_val)

        # find distance of the single test example w.r.t. all the training examples
        distances = self.euclidean_dist(norm_test_single, norm_train)

        # find the nearest neighbors
        nn_indices = self.find_k_nearest_neighbors(k, distances)

        # find the labels of the nearest neighbors
        neighbor_labels = training_labels[nn_indices]

        best_label = self.predict_label(neighbor_labels)

        return best_label

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = np.apply_along_axis(self.kNN_one_example, 1, self.training_data, self.training_data, self.training_labels, self.k)

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        print("test in predict")
        test_labels = np.apply_along_axis(self.kNN_one_example, 1, test_data, self.training_data, self.training_labels, self.k)

        return test_labels
