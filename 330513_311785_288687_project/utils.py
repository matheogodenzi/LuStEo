import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def label_to_onehot(label):
    one_hot_labels = np.zeros([label.shape[0], int(np.max(label)+1)])
    one_hot_labels[np.arange(label.shape[0]), label.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    return np.argmax(onehot, axis=1)

def normalize_fn(data, means, stds):
    """This function takes the data, the means,
    and the standard deviatons(precomputed). It
    returns the normalized data.

    Inputs:
        data : shape (NxD)
        means: shape (1XD)
        stds : shape (1xD)

    Outputs:
        data_normed: shape (NxD)
    """
    # WRITE YOUR CODE HERE
    # return the normalized features
    return (data - means) / stds


def plotting_param(file_name):
    df = pd.read_csv(file_name)
    del df['Unnamed: 0']
    df.plot(kind = 'scatter',
        x = df.columns[0],
        y = df.columns[1],
        color = 'red')
    plt.xscale('log')
    plt.show()
    return

def plotting_it(file_name):
    df = pd.read_csv(file_name)
    df[df.columns[1]] = df[df.columns[1]]/100
    fig, ax = plt.subplots()
    #ax.plot(df[df.columns[0]], df[df.columns[1]], color = 'red')
    #ax.set_xlabel("iterations")
    #ax.set_ylabel("acc (%)")
    #ax2 = ax.twinx()
    #ax2.plot(df[df.columns[0]], df[df.columns[2]], df[df.columns[3]], color='blue')
    #ax2.set_ylabel("MF1 & MSE")
    df.plot(x=df.columns[0], y=[df.columns[1], df.columns[2], df.columns[3]])
    plt.show()
    return

#if __name__ == "__main__":
    #plotting_param("param_opt.csv")
    #plotting_it("metrics_it.csv")
