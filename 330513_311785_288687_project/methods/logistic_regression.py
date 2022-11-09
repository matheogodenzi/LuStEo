import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot
from utils import onehot_to_label
from metrics import accuracy_fn


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
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
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """

        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        elif len(args) > 0:
            self.lr = args[0]
        else:
            self.lr = 0.1

        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) > 1:
            self.max_iters = args[1]
        else:
            self.max_iters = 10

    def f_softmax(self, training_data, w):
        """ Softmax function

        Args:
            data (np.array): Input data of shape (N, D)
            w (np.array): Weights of shape (D, C) where C is # of classes

        Returns:
            res (np.array): Probabilites of shape (N, C), where each value is in
                range [0, 1] and each row sums to 1.
        """
        f_softmax = np.zeros((training_data.shape[0], w.shape[1]))

        for i in range(training_data.shape[0]):
            norm_fact = np.sum(np.exp(w.T@training_data[i,:]))
            for j in range(w.shape[1]):
                f_softmax[i,j] = np.exp(w.T[j, :]@training_data[i,:])/norm_fact

        '''
        a = round(np.sum(f_softmax[i, :]),2)
        print(a)
        '''
        res = f_softmax
        return res

    def gradient_logistic_multi(self, training_data, training_labels, w):
        """ Gradient function for multi class logistic regression

        Args:
            data (np.array): Input data of shape (N, D)
            labels (np.array): Labels of shape  (N, C)  (in one-hot representation)
            w (np.array): Weights of shape (D, C)

        Returns:
            grad_w (np.array): Gradients of shape (D, C)
        """
        # print("f_soft :", self.f_softmax(training_data, w).shape)
        # print("training_labels",training_labels.shape)
        grad_w = training_data.T@(self.f_softmax(training_data, w)-label_to_onehot(training_labels))

        return grad_w

    def logistic_regression_classify_multi(self, training_data, w):
        """ Classification function for multi class logistic regression.

        Args:
            data (np.array): Dataset of shape (N, D).
            w (np.array): Weights of logistic regression model of shape (D, C)
        Returns:
            predictions (np.array): Label assignments of data of shape (N, ) (NOT one-hot!).
        """
        # YOUR CODE HERE: find predictions, argmax to find the correct label
        y_hat = self.f_softmax(training_data, w)
        predictions = np.argmax(y_hat, 1)
        return predictions

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """

        # print(label_to_onehot(training_labels).shape)
        self.w = np.random.normal(0, 0.1, [training_data.shape[1], label_to_onehot(training_labels).shape[1]])
        for it in range(self.max_iters):
            self.w = self.w - self.lr*self.gradient_logistic_multi(training_data, training_labels, self.w)

            # print("training data shape : ", training_data.shape)
            # print("w : ", self.w.shape)
            pred_labels = self.logistic_regression_classify_multi(training_data, self.w)
            if accuracy_fn(pred_labels, training_labels) == 1:
                break

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        # print("predicting values...")
        test_labels = self.logistic_regression_classify_multi(test_data, self.w)
        return test_labels
