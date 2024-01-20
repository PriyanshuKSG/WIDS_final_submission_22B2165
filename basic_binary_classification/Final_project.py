import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANN_Priyanshu:

    def __init__(self, learning_rate = 0.001, num_iterations = 10):
        # Just to verify the cration of the model
        print("Model created successfully!")

        # Initialising the learning rate and number of iterations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def cost_fn(self,A_L, Y):
        epsilon = 1e-15
        A_L = np.clip(A_L, epsilon, 1 - epsilon)
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A_L + epsilon))
        return cost
    
    def binary_loss(self, A_L, y):
        epsilon = 1e-15  # Small epsilon value to avoid numerical instability
        A_L = np.clip(A_L, epsilon, 1 - epsilon)
        m = y.shape[1]
        loss = -1 / m * np.sum(y * np.log(A_L) + (1 - y) * np.log(1 - A_L))
        return loss
    
    def sigmoid_af(self, Z):
        denom = 1 + np.exp(-Z)
        A = 1/denom
        cache = Z
        return A, cache
    
    def relu_af(self, Z):
        cache = Z
        return np.maximum(0, Z), cache
    
    def softmax_af(self, Z):
        e_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Subtracting max for numerical stability
        A = e_x / np.sum(e_x, axis=1, keepdims=True)
        cache = Z
        return A, cache
    
    def relu_derivative(self, dA, cache):
        Z = cache
        dZ = np.zeros(Z.shape)
        dZ[Z > 0] = 1
        assert (dZ.shape == Z.shape)
        return dZ
    
    def sigmoid_derivative(self, dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = np.zeros(Z.shape)
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def softmax_derivative(self, dA, cache):
        Z = cache
        s = self.softmax_af(Z)
        return dA * (s - s**2)
    
    def initialize_parameters(self, num_nodes_in_layers):

        print("Initialization function")
        L = len(num_nodes_in_layers)

        # parameters is a dictionary from where we'll extract all our parameters W and b
        parameters = {}

        # Convention: W1 is the weight matrix of first hidden layer. b2 is the bias matrix of second hidden layer. Similarly for other layers.

        # Xavier Initialization
        for l in range(1,L):
            parameters['W' + str(l)] = np.random.randn(num_nodes_in_layers[l], num_nodes_in_layers[l-1]) / (np.sqrt(num_nodes_in_layers[l-1]))
            parameters['b' + str(l)] = np.zeros((num_nodes_in_layers[l], 1))

        print("Number of layers = ", len(parameters) // 2)
        print("-----------------------------------------------------------------------")
        return parameters
    
    def forward_linear(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        return Z, cache
    
    def forward_activation(self, A_prev, W, b, activation = "relu"):

        Z, linear_cache = self.forward_linear(A_prev, W, b)

        if activation == "relu":
            A, activation_cache = self.relu_af(Z)
        elif activation == "sigmoid":
            A, activation_cache = self.sigmoid_af(Z)
        elif activation == "softmax":
            A, activation_cache = self.softmax_af(Z)

        cache = (linear_cache, activation_cache)

        assert(A.shape == Z.shape)

        return A, cache
    
    def forward_prop(self, X, parameters):

        print("Forward propagation funtion")
        # L = number of layers
        # L = 4
        L = len(parameters) // 2
        cachess = []
        A = X

        for l in range(0, L-1):
            print("Layer ", l+1, " done")
            A_prev = A
            A, cache = self.forward_activation(A_prev, parameters["W" + str(l+1)], parameters["b" + str(l+1)])
            cachess.append(cache)

        print("Layer ", L, " done")
        A_L, cache = self.forward_activation(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "softmax")
        cachess.append(cache)

        #assert(A_L.shape == (10,60000))
        print("-----------------------------------------------------------------------")
        return A_L, cachess
    
    def linear_backward(self, dZ, cache):
        m = dZ.shape[1]
        A_prev, W, b = cache
        dW = (1/m)*(np.dot(dZ, A_prev.T))
        db = (1/m)*(np.sum(dZ, axis = 1, keepdims = True))
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def backward_activation(self, dA, cache, activation = "relu"):

        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_derivative(dA, activation_cache)
        elif activation == "softmax":
            dZ = self.softmax_derivative(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_derivative(dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def backward_prop(self, A_L, Y, caches):

        print("Backward propagation funtion")

        grads = {}

        L = len(caches)
        m = A_L.shape[1]

        current_cache = caches[-1]
        dA_L = - ((np.divide(Y, A_L)) - (np.divide(1-Y, 1-A_L)))

        print("Layer ", L, " done")
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.backward_activation(dA_L, current_cache, activation = "sigmoid")

        for l in reversed(range(L-1)):
            print("Layer ", l+1, " done")
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backward_activation(grads["dA" + str(l + 2)], current_cache, activation = "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    def update_parameters(self, parameters, grads, learning_rate):

        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate*grads["db" + str(l+1)]

        return parameters
    
    def cost_plot(self, cost):

        x = list(range(1, len(cost) + 1))

        plt.plot(x, cost, 0., color = 'b')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    
    def fit(self, X, Y, num_nodes_in_layers):

        print("Training in progres............")

        costs = np.zeros(self.num_iterations)

        parameters = self.initialize_parameters(num_nodes_in_layers)

        for i in range(self.num_iterations):
            print("Iteration ", i+1)
            A_L, cache = self.forward_prop(X, parameters)
            #y_pred = np.argmax(A_L, axis = 0)
            y_pred = (A_L >= 0.5).astype(int)
            cost = self.binary_loss(y_pred, Y)
            grads = self.backward_prop(A_L, Y, cache)
            parameters = self.update_parameters(parameters, grads, self.learning_rate)
            costs[i] = cost
            print("---------------------------------------------------------------------")

        print("Model trained successfully!")
        self.cost_plot(costs)
        return parameters
    
    def predict(self, X, parameters):
        Y_pred, cache = self.forward_prop(X, parameters)
        ans = (Y_pred >= 0.5).astype(int)
        return ans
    
    def accuracy(self, y, y_pred):
        m = y.shape[1]
        correct_predictions = np.sum(y == y_pred)
        acc = correct_predictions / m
        return acc

#########################################################################################
    
"""
path = "C:\\Users\\Priyanshu\\OneDrive\\Desktop\\WIDS\\diabetes (1).csv"
df = pd.read_csv(path)
#print(df.head(10)) 
Y = df['Outcome']
X = df.drop('Outcome', axis = 1)
#print(Y.value_counts())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 97)
#print(X_train.shape, Y_test.shape)
scaler = StandardScaler()
X_new_train = scaler.fit_transform(X_train)
X_new_test = scaler.transform(X_test)
X_new_train = X_new_train.reshape(8, 614)
Y_train = np.array(Y_train).reshape(1,614)
X_new_test = X_new_test.reshape(8, 154)
Y_test = np.array(Y_test).reshape(1,154)
#print(X_new_train.shape, Y_test.shape)
m = X_train.shape[0]
#print(m)
model = ANN_Priyanshu()
params = model.fit(X_new_train, Y_train, [8, 64, 32, 16, 8, 4, 2, 1])
Y_pred_train = model.predict(X_new_train, params)
print("Training accuracy = ", model.accuracy(Y_train, Y_pred_train)*100,"%")
Y_pred_test = model.predict(X_new_test, params)
print("Testing accuracy = ",model.accuracy(Y_test, Y_pred_test)*100,"%")
"""

########################################################################################