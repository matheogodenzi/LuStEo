import numpy as np

class PCA(object):
    """
        PCA dimensionality reduction object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, find_principal_components, and reduce_dimension work correctly.
    """
    def __init__(self, *args, **kwargs):
        """
            You don't need to initialize the task kind for PCA.
            Call set_arguments function of this class.
        """
        self.set_arguments(*args, **kwargs)
        #the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        #the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The PCA class should have a variable defining the number of dimensions (d).
            You can either pass this as an arg or a kwarg.
        """

        if "d" in kwargs:
           self.d = kwargs["d"]
        elif len(args) > 0:
            self.d = args[0]


    def find_principal_components(self, training_data):
        """
            Finds the principal components of the training data. Returns the explained variance in percentage.
            IMPORTANT:
            This function should save the mean of the training data and the principal components as
            self.mean and self.W, respectively.

            Arguments:
                training_data (np.array): training data of shape (N,D)
            Returns:
                exvar (float): explained variance
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        # Compute the mean of data
        self.mean = np.mean(training_data, 0)
        # Center the data with the mean
        X_tilde = training_data - self.mean

        
        # Create the covariance matrix
        C = 1/X_tilde.shape[0]*(X_tilde.T@X_tilde)

        # Compute the eigenvectors and eigenvalues. 
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Choose the top d eigenvalues and corresponding eigenvectors. Sort the eigenvalues( with corresponding eigenvectors )
        # in decreasing order first.
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, 1)
        
        # Create matrix W and the corresponding eigen values
        self.W = eigvecs[:self.d].T
        
        eg = eigvals[:self.d]

        
        
        # Compute the explained variance
        exvar = np.sum(eg)/np.sum(eigvals)*100

        return exvar

        
    def reduce_dimension(self, data):
        """
            Reduce the dimensions of the data, using the previously computed
            self.mean and self.W.

            Arguments:
                data (np.array): data of shape (N,D)
            Returns:
                data_reduced (float): reduced data of shape (N,d)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        X_tilde_data = data - self.mean
        data_reduced = X_tilde_data@self.W

        return data_reduced
