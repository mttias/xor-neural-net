import numpy as np
import matplotlib.pyplot as plt

# Constants
INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS = 2, 2, 1
LEARNING_RATE = 0.1
ITERATIONS = 10000
SEED = 42

# Define the inputs
a = np.array([0, 0, 1, 1])
b = np.array([0, 1, 0, 1])
y_xor = np.array([[0, 1, 1, 0]])

total_input = np.array([a, b])
samples = total_input.shape[1]

# Functions for the neural network
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

activation_func = sigmoid

def initialize_weights():
    np.random.seed(SEED)
    w1 = np.random.rand(HIDDEN_NEURONS, INPUT_NEURONS)
    w2 = np.random.rand(OUTPUT_NEURONS, HIDDEN_NEURONS)
    return w1, w2

def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = activation_func(z1)
    z2 = np.dot(w2, a1)
    a2 = activation_func(z2)
    return z1, a1, z2, a2

def back_prop(m, w1, w2, z1, a1, z2, a2, y):
    dz2 = a2-y
    dw2 = np.dot(dz2, a1.T)/m
    dz1 = np.dot(w2.T, dz2) * a1*(1-a1)
    dw1 = np.dot(dz1, total_input.T)/m
    dw1 = np.reshape(dw1, w1.shape)
    dw2 = np.reshape(dw2, w2.shape)
    return dz2, dw2, dz1, dw1

def train_network():
    w1, w2 = initialize_weights()
    losses = []

    for _ in range(ITERATIONS):
        z1, a1, z2, a2 = forward_prop(w1, w2, total_input)
        # cross-entropy loss function
        # the task was is perform a binary classification (XOR operation),
        # and the output is a probability (since it was produced by a sigmoid activation function),
        # which is why binary cross-entropy loss is a suitable choice.
        # If the output is a continuous value or the task was multi-class classification,
        # a different loss function would likely be more appropriate.
        loss = -(1/samples)*np.sum(y_xor*np.log(a2)+(1-y_xor)*np.log(1-a2))
        losses.append(loss)
        da2, dw2, dz1, dw1 = back_prop(samples, w1, w2, z1, a1, z2, a2, y_xor)
        w2 = w2-LEARNING_RATE*dw2
        w1 = w1-LEARNING_RATE*dw1
    
    return w1, w2, losses

def predict(w1, w2, input):
    z1, a1, z2, a2 = forward_prop(w1, w2, input)
    a2 = np.squeeze(a2)

    if a2 >= 0.5:
        print("For input", [i[0] for i in input], "output is 1")
    else:
        print("For input", [i[0] for i in input], "output is 0")

# Running the code
w1, w2, losses = train_network()

# Display losses
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

# Test the network
test_inputs = [[0,0],[0,1],[1,0],[1,1]]
for test_input in test_inputs:
    predict(w1, w2, np.array([test_input]).T)
