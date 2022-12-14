import numpy as np
import argparse
import utils
import time


# these will be imported in MS2. uncomment then!
import torch
from torch.utils.data import DataLoader
from methods.deep_network import SimpleNetwork, Trainer

from data import H36M_Dataset, FMA_Dataset, Movie_Dataset
from methods.pca import PCA
from methods.cross_validation import cross_validation
from metrics import accuracy_fn,mse_fn, macrof1_fn
from methods.knn import KNN
from methods.dummy_methods import DummyClassifier, DummyRegressor
from methods.logistic_regression import LogisticRegression
from methods.linear_regression import LinearRegression

def main(args):
    # First we create all of our dataset objects. The dataset objects store the data, labels (for classification) and the targets for regression
    if args.dataset=="h36m":
        train_dataset = H36M_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = H36M_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

        ''' contest test dataset'''
        # test_dataset = H36M_Dataset(split="test2", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        val_dataset = H36M_Dataset(split="val",path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

    elif args.dataset=="music":
        train_dataset = FMA_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = FMA_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        val_dataset = FMA_Dataset(split="val",path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

    elif args.dataset=="movies":
        train_dataset = Movie_Dataset(split="train", path_to_data=args.path_to_data)
        test_dataset = Movie_Dataset(split="test", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)
        #uncomment for MS2
        val_dataset = Movie_Dataset(split="val", path_to_data=args.path_to_data, means=train_dataset.means, stds=train_dataset.stds)

    # Note: We only use the following methods for more old-school methods, not the nn!
    train_data, train_regression_target, train_labels = train_dataset.data, train_dataset.regression_target, train_dataset.labels
    test_data, test_regression_target, test_labels = test_dataset.data, test_dataset.regression_target, test_dataset.labels
    #print(train_labels)
    # print(train_labels)
    print("Dataloading is complete!")

    #initializing the time calulation
    s1 = time.time()

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d = 200)
        pca_obj.find_principal_components(train_data)
        train_data = pca_obj.reduce_dimension(train_data)
        #train_regression_target = pca_obj.reduce_dimension(train_regression_target)
        test_data = pca_obj.reduce_dimension(test_data)
        #test_regression_target = pca_obj.reduce_dimension(test_regression_target)

    # Neural network. (This part is only relevant for MS2.)
    if args.method_name == "nn":
        # Pytorch dataloaders
        print("Using deep network")
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # determining the feature dimensions and number of classes of the training dataset
        '''
        print(f'feature dimensions of the training dataset are : {train_dataset.feature_dim}')
        print(f'number of classes to sort the data into : {train_dataset.num_classes}')
        for it, batch in enumerate(train_dataloader):
            print(batch[2])
        '''

        # create model
        model = SimpleNetwork(input_size=train_dataset.feature_dim, num_classes=train_dataset.num_classes, hidden_size=args.hidden_layers)

        # training loop
        trainer = Trainer(model, lr=args.lr, epochs=args.max_iters)
        trainer.train_all(train_dataloader, val_dataloader, test_dataloader)
        results_class = trainer.eval(test_dataloader)
        print('shape = ',results_class.shape)
        '''printing accuracies'''
        #print(f'final validation accuracy {trainer.acc_val}')
        #print(f'final validation f1 score {trainer.f1_val}')
        #print(results_class.size())

        #torch.save(results_class, "results_class.txt")
        np.savetxt('results.txt', results_class)
        np.save("results_class", results_class.numpy())

    # classical ML methods (MS1 and MS2)
    # we first create the classification/regression objects
    # search_arg_vals and search_arg_name are defined for cross validation
    # we show how to create the objects for DummyClassifier and DummyRegressor
    # the rest of the methods are up to you!
    else:
        if args.method_name == "dummy_classifier":
            method_obj =  DummyClassifier()
            search_arg_vals = [1,2,3]
            search_arg_name = "dummy_arg"

        elif args.method_name == 'dummy_regressor':
            method_obj = DummyRegressor()
            search_arg_vals = [1,2,3]
            train_labels = train_regression_target
            search_arg_name = "dummy_arg"

        elif args.method_name == "ridge_regression":
            method_obj = LinearRegression(lmda=args.ridge_regression_lmda)

            """
            # defining bias term training data
            ones_column = np.ones(train_data.shape[0]).reshape(train_data.shape[0],-1)
            train_data = np.concatenate((ones_column, train_data), axis=1)

            # defining bias term testing data
            ones_column = np.ones(test_data.shape[0]).reshape(test_data.shape[0],-1)
            test_data = np.concatenate((ones_column, test_data), axis=1)
            """

            # getting the targets instead of all the labels
            train_labels = train_regression_target

            # for cross validation
            search_arg_vals = [1,50, 75, 100, 150, 500, 1000, 1500, 2000]
            search_arg_name = "lmda"

        elif args.method_name == "logistic_regression":
            method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
            search_arg_vals = [1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            search_arg_name = "lr"

        elif args.method_name == "knn":
            method_obj = KNN(k=args.knn_neighbours)
            search_arg_vals = [6,3,4,1] #6 best with best accuracy 0.940566769143822
            search_arg_name = "knn_neighbours"


        # cross validation (MS1)
        if args.use_cross_validation:
            print("Using cross validation")
            best_arg, best_val_acc = cross_validation(method_obj=method_obj, search_arg_name=search_arg_name, search_arg_vals=search_arg_vals, data=train_data, labels=train_labels, k_fold=4)
            # set the classifier/regression object to have the best hyperparameter found via cross validation:
            method_obj.set_arguments(best_arg)
            print(f"best_arg :{best_arg} \nbest_val_acc : {best_val_acc}")

        # FIT AND PREDICT:
        method_obj.fit(train_data, train_labels)
        pred_labels = method_obj.predict(test_data)
        # Report test results
        if method_obj.task_kind == 'regression':
            loss = mse_fn(pred_labels,test_regression_target)
            print("Final loss is", loss)
        else:
            acc = accuracy_fn(pred_labels,test_labels)
            print("Final classification accuracy is", acc)
            macrof1 = macrof1_fn(pred_labels,test_labels)
            print("Final macro F1 score is", macrof1)

    #calculating the time of the process
    s2 = time.time()
    print(f"{args.method_name} takes {s2-s1} seconds")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="h36m", type=str, help="choose between h36m, movies, music")
    parser.add_argument('--path_to_data', default="..", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ..")
    parser.add_argument('--method_name', default="knn", type=str, help="knn / logistic_regression / nn")
    parser.add_argument('--knn_neighbours', default=3, type=int, help="number of knn neighbours")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--ridge_regression_lmda', type=float, default=1, help="lambda for ridge regression")
    parser.add_argument('--max_iters', type=int, default=1000, help="max iters for methods which are iterative")
    parser.add_argument('--use_cross_validation', action="store_true", help="to enable cross validation")

    # Feel free to add more arguments here if you need
    parser.add_argument('--hidden_layers', default=(52, 42, 32, 22), help="sets the number of hidden neurons in the deep network on both layers")

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    args = parser.parse_args()
    main(args)
