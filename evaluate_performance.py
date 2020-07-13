
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('loss_scores', 'rb') as f:
    loss_scores = pickle.load(f)

n_epochs = range(np.shape(loss_scores)[1])
train_loss = loss_scores[0]
test_loss = loss_scores[1]

plt.plot(n_epochs, train_loss)
plt.plot(n_epochs, test_loss)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(['train error', 'validation error'])
plt.show()