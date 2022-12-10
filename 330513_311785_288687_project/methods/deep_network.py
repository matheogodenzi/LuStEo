import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy_fn, macrof1_fn
import numpy as np

## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes, hidden_size=(300, 20)):
        super(SimpleNetwork, self).__init__()

        # defining characteristic parameters of the network
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # defining layers transitions
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_classes)
        # self.fc4 = nn.Linear(hidden_size[2], num_classes)
    def forward(self, x):
        """
        Takes as input the data x and outputs the
        classification outputs.
        Args:
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """
        output_class = F.relu(self.fc1(x))
        output_class = F.relu(self.fc2(output_class))
        output_class = self.fc3(output_class)
        #output_class = self.fc4(x)

        return output_class

class Trainer(object):

    """
        Trainer class for the deep network.
    """

    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta

        self.classification_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            self.eval(dataloader_val)

            if (ep+1) % 50 == 0:
                print("Reduce Learning rate")
                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.8


    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """

        # setting my model to training mode
        self.model.train()

        # looping over each batch of the dataloader
        for it, batch in enumerate(dataloader):
            # Load a batch, break it down in samples and targets.
            x, y, y_class = batch

            # Run forward pass.
            logits = self.model.forward(x)  # YOUR CODE HERE

            # Compute loss (using 'criterion').
            loss = self.classification_criterion(logits, y_class)  # YOUR CODE HERE

            # Run backward pass.
            loss.backward()  # YOUR CODE HERE

            # Update the weights using optimizer.
            self.optimizer.step()  # YOUR CODE HERE

            # Zero-out the accumulated gradients.
            self.optimizer.zero_grad()  # YOUR CODE HERE


    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()

            Returns:
                Note: N is the amount of validation/test data.
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        # initiating evaluation mode
        self.model.eval()

        with torch.no_grad():
            acc_run = 0
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x, y, y_class = batch
                curr_bs = x.shape[0]
                res = torch.argmax(self.model(x),dim=1)
                acc_run += accuracy_fn(res.numpy(),y_class.numpy())*curr_bs

                # verifying shapes
                #print(it)
                #print('res', res)
                #print('y_class', y_class)
                #print('********')

                if it == 0:
                    results_class = res
                    y_tot_class = y_class
                else:
                    results_class = torch.cat((results_class, res))
                    y_tot_class = torch.cat((y_tot_class, y_class))
                #print(it)
            self.acc = acc_run/len(dataloader.dataset)
            self.f1 = macrof1_fn(results_class.numpy(), y_tot_class.numpy())
            print('acc =', self.acc)
            print('f1 score =', self.f1)

        return results_class
