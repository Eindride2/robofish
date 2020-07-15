import numpy as np
import matplotlib.pyplot as plt
import pickle
from hyper_params import *

def plot_scores(scores, load_from_file, filename):

    if load_from_file:
        with open(filename, 'rb') as f:
            scores = pickle.load(f)

    n_epochs = range(epochs)
    train_loss = scores[0]
    test_loss = scores[1]
    confidence_turn = scores[2]
    confidence_speed = scores[3]
    accuracy_turn = scores[4]
    accuracy_speed = scores[5]

    plt.subplot(131)
    plt.plot(n_epochs, train_loss)
    plt.plot(n_epochs, test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['train error', 'validation error'])
    plt.subplot(132)
    plt.plot(n_epochs, confidence_turn)
    plt.plot(n_epochs, confidence_speed)
    plt.legend(['angular turn', 'linear speed'])
    plt.xlabel('Epochs')
    plt.ylabel('Confidence')
    plt.subplot(133)
    plt.plot(n_epochs, accuracy_turn)
    plt.plot(n_epochs, accuracy_speed)
    plt.legend(['angular turn', 'linear speed'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

#plot_scores(scores = None, load_from_file = True, filename = 'scores')