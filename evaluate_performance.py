
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_scores(scores, load_from_file, filename):

    if load_from_file:
        with open(filename, 'rb') as f:
            scores = pickle.load(f)

    n_epochs = range(np.shape(scores)[1])
    train_loss = scores[0]
    test_loss = scores[1]
    confidence_turn = scores[2]
    confidence_speed = scores[3]

    plt.subplot(121)
    plt.plot(n_epochs, train_loss)
    plt.plot(n_epochs, test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['train error', 'validation error'])
    plt.subplot(122)
    plt.plot(n_epochs, confidence_turn)
    plt.plot(n_epochs, confidence_speed)
    plt.legend(['confidence angular turn', 'confidence linear speed'])
    plt.xlabel('Epochs')
    plt.ylabel('Confidence')
    plt.show()

#plot_scores(scores = None, load_from_file = True, filename = 'scores')