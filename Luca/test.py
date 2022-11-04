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

        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        # if not, then check if args is a list with size bigger than 0.
        elif "lr" in kwargs:
            self.lr = kwargs["lr"]
        elif len(args) > 0:
            self.max_iters = args[5]
            self.lr = args[7]

        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
       

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """
        
        
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##
        def f_softmax(data, w):
            """ Softmax function

            Args:
                data (np.array): Input data of shape (N, D)
                w (np.array): Weights of shape (D, C) where C is # of classes

            Returns:
                res (np.array): Probabilites of shape (N, C), where each value is in
                    range [0, 1] and each row sums to 1.
            """
            N = data.shape[0]
            C = w.shape[1]

            s_exp = np.exp(data @ w)
            s_exp_sum = np.sum(s_exp, axis=1)

            res = np.zeros((data.shape[0], w.shape[1]))
            for i in range(0, N):
                res[i] = s_exp[i] / s_exp_sum[i]
            # We've left this part for you to code for your projects.
            return res






        def loss_logistic_multi(data, labels, w):
            """ Loss function for multi class logistic regression

            Args:
                data (np.array): Input data of shape (N, D)
                labels (np.array): Labels of shape  (N, C) (in one-hot representation)
                w (np.array): Weights of shape (D, C)

            Returns:
                float: Loss value
            """
            # We've left this part for you to code for your projects.
            loss = -np.sum(np.sum(labels * np.log(f_softmax(data, w))))
            return loss





        def gradient_logistic_multi(data, labels, w):
            """ Gradient function for multi class logistic regression

            Args:
                data (np.array): Input data of shape (N, D)
                labels (np.array): Labels of shape  (N, )
                w (np.array): Weights of shape (D, C)

            Returns:
                grad_w (np.array): Gradients of shape (D, C)
            """
            # We've left this part for you to code for your projects.
            grad_w = data.T @ (f_softmax(data, w) - labels)
            return grad_w




        def logistic_regression_classify_multi(data, w):
            """ Classification function for multi class logistic regression.

            Args:
                data (np.array): Dataset of shape (N, D).
                w (np.array): Weights of logistic regression model of shape (D, C)
            Returns:
                np. array: Label assignments of data of shape (N, ).
            """
            #### write your code here: find predictions, argmax to find the correct label
            # We've left this part for you to code for your projects.
            predictions = np.argmax(f_softmax(data, w), axis=1)

            return predictions



        def logistic_regression_train_multi(data, labels, k=3, max_iters=10, lr=0.001,
                                            print_period=5, plot_period=5):
            """ Classification function for multi class logistic regression.

            Args:
                data (np.array): Dataset of shape (N, D).
                labels (np.array): Labels of shape (N, C)
                k (integer): Number of classes. Default=3
                max_iters (integer): Maximum number of iterations. Default:10
                lr (integer): The learning rate of  the gradient step. Default:0.001
                print_period (int): Num. iterations to print current loss.
                    If 0, never printed.
                plot_period (int): Num. iterations to plot current predictions.
                    If 0, never plotted.

            Returns:
                np. array: Label assignments of data of shape (N, ).
            """
            weights = np.random.normal(0, 0.1, [data.shape[1], k])
            for it in range(max_iters):
                # YOUR CODE HERE
                gradient = gradient_logistic_multi(data, labels, weights)
                weights = weights - lr * gradient
                ##################################
                predictions = logistic_regression_classify_multi(data, weights)
                if accuracy_fn(helpers.onehot_to_label(labels), predictions) == 1:
                    break
                # logging and plotting
                if print_period and it % print_period == 0:
                    print('loss at iteration', it, ":", loss_logistic_multi(data, labels, weights))
                if plot_period and it % plot_period == 0:
                    fig = helpers.visualize_predictions(data=data, labels_gt=helpers.onehot_to_label(labels),
                                                        labels_pred=predictions, title="iteration " + str(it))
            fig = helpers.visualize_predictions(data=data, labels_gt=helpers.onehot_to_label(labels),
                                                labels_pred=predictions, title="final model")
            return weights

        # In[113]:

        weights_multi = logistic_regression_train_multi(training_data, training_labels, max_iters=20, lr=1e-3, print_period=5,
                                                        plot_period=5)



        predictions_multi = logistic_regression_classify_multi(data_test, weights_multi)
        fig = helpers.visualize_predictions(data=data_test, labels_gt=helpers.onehot_to_label(labels_test),
                                            labels_pred=predictions_multi, title="test result")
        print("Accuracy is", accuracy_fn(helpers.onehot_to_label(labels_test), predictions_multi))


        self.D, self.C = training_data.shape[1], int(np.max(training_labels) + 1)
        pred_labels = predictions_multi
