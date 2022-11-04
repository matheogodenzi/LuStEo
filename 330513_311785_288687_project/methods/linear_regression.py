import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object.
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
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
        self.task_kind = "regression"
        self.set_arguments(*args, **kwargs)


    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            In case of ridge regression, you need to define lambda regularizer(lmda).

            You can either pass these as args or kwargs.
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        # first checks if "lmda" was passed as a kwarg.
        if "lmda" in kwargs:
            self.lmda = kwargs["lmda"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) > 0 :
            self.lmda = args[0]
        # if there were no args or kwargs passed, we set the lmda to 0 (default value).
        else:
            self.lmda = 1


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): predicted target of shape (N,regression_target_size)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        ones_column = np.ones(training_data.shape[0]).reshape(training_data.shape[0],-1)
        X_bias = np.concatenate((training_data, ones_column), axis=1)
        Identity = np.identity(X_bias.shape[1])
        self.w = np.linalg.inv((X_bias.T@X_bias) + self.lmda*Identity)@X_bias.T@training_labels
        return X_bias@self.w



    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): predicted targets of shape (N,regression_target_size)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        ones_column = np.ones(test_data.shape[0]).reshape(test_data.shape[0],-1)
        X_test_bias = np.concatenate((test_data, ones_column), axis=1)
        return X_test_bias@self.w
