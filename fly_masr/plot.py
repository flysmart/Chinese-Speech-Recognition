import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    train_losses = np.load("train_losses.npy")
    valid_losses = np.load("valid_losses.npy")
    train_losses1 = np.load("train_losses1.npy")
    valid_losses1 = np.load("valid_losses1.npy")

    f1 = plt.figure('train_losses')
    plt.plot(train_losses, color='red', label='Transformer')
    plt.plot(train_losses1, color='blue',label='LAS')
    plt.xlabel("epoch")
    plt.ylabel("train_losses")
    plt.legend()
    f2 = plt.figure('valid_losses')
    plt.plot(valid_losses, color='red',label='Transformer')
    plt.plot(valid_losses1, color='blue',label='LAS')
    plt.xlabel("epoch")
    plt.ylabel("valid_losses")
    plt.legend()
    plt.show()
