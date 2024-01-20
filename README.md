# Artificial Neural Network (ANN) Implementation from Scratch in Python

## Overview

This project is a Python implementation of an Artificial Neural Network (ANN) built from scratch using Object-Oriented Programming (OOP) principles. The purpose of this implementation is to gain a deeper understanding of the inner workings of neural networks by constructing the fundamental components from the ground up.

## Features

- **Feedforward Neural Network:** The implemented neural network supports feedforward operations, allowing it to make predictions on input data.
- **Backpropagation:** Backpropagation is used for training the neural network, optimizing the weights to minimize the error between predicted and actual outputs.
- **Activation Functions:** You can choose from various activation functions, including Sigmoid, Softmax, or ReLU, for the hidden layers.
- **Customizable Architecture:** The number of layers, neurons per layer, and activation functions are customizable to suit different tasks and datasets.
- **Gradient Descent:** Batch Gradient Descent (BGD) is employed as the optimization algorithm.
- **Cost function:** Binary Cross Entropy Loss function was used due to its convex nature so that our Gradient Descent optimization algorithm can effectively converge to the minima.

## Findings and Research

### Forward Propagation at any layer \(l\):

![Forward Propagation Equations](https://latex.codecogs.com/svg.latex?Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]})

![Activation Equations](https://latex.codecogs.com/svg.latex?A^{[l]}=g^{[l]}(Z^{[l]}))

Where:
- ![Activation Equations](https://latex.codecogs.com/svg.latex?Z^{[l]}) is the weighted sum for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?W^{[l]}) is the weight matrix for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?A^{[l-1]}) is the activation from the previous layer,
- ![Activation Equations](https://latex.codecogs.com/svg.latex?b^{[l]}) is the bias for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?g^{[l]}) is the activation function for layer \(l\).

### Backward Propagation at any layer \(l\):

![dZ Equations](https://latex.codecogs.com/svg.latex?dZ^{[l]}=\frac{\partial\mathcal{L}}{\partial{Z^{[l]}}}=dA^{[l]}\cdot{g'^{[l]}(Z^{[l]})})

![dW Equations](https://latex.codecogs.com/svg.latex?dW^{[l]}=\frac{\partial\mathcal{L}}{\partial{W^{[l]}}}=\frac{1}{m}dZ^{[l]}\cdot{A^{[l-1]T}})

![db Equations](https://latex.codecogs.com/svg.latex?db^{[l]}=\frac{\partial\mathcal{L}}{\partial{b^{[l]}}}=\frac{1}{m}\sum_{i=1}^{m}dZ^{[l](i)})

![dA Equations](https://latex.codecogs.com/svg.latex?dA^{[l-1]}=\frac{\partial\mathcal{L}}{\partial{A^{[l-1]}}}=W^{[l]T}\cdot{dZ^{[l]}})

Where:
- ![Activation Equations](https://latex.codecogs.com/svg.latex?dZ^{[l]}) is the gradient of the cost with respect to the weighted sum for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?dW^{[l]}) is the gradient of the cost with respect to the weights for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?db^{[l]}) is the gradient of the cost with respect to the biases for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?dA^{[l-1]}) is the gradient of the cost with respect to the activation from the previous layer,
- ![Activation Equations](https://latex.codecogs.com/svg.latex?g'^{[l]}) is the derivative of the activation function for layer \(l\),
- ![Activation Equations](https://latex.codecogs.com/svg.latex?m) is the number of examples in the training set.

### Comparison of Model performance:

- **Training Accuracy:**
  - **Logistic Regression (sklearn):** 77%
  - **Custom-built ANN:** 65.63%

- **Testing Accuracy:**
  - **Logistic Regression (sklearn):** 78%
  - **Custom-built ANN:** 63%

There could be various reasons for the observed differences in accuracy:
1. **Model Complexity:** Logistic regression is a simpler model, suitable for linearly separable problems. If the data is not well-suited for the complexity of a neural network, logistic regression may outperform it.
2. **Data Size:** The size of the dataset can influence model performance. Smaller datasets may lead to better generalization for simpler models like logistic regression.
3. **Hyperparameter Tuning:** Absence of hyperparameter tuning such as learning rate, architecture etc. might also be the reason.

## Dependencies

- Python 3.x
- NumPy: `pip install numpy`

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/PriyanshuKSG/WIDS_final_submission_22B2165.git
   cd WIDS_final_submission_22B2165
2. Import the ANN Priyanshu class into your Python script or Jupyter Notebook:
   ```code
   from basic_binary_classification import ANN_Priyanshu
3. Create an instance of the ANN Priyanshu class and configure the architecture:
   ```code
   model = ANN_Priyanshu()
4. Train the neural network on your dataset:
   ```code
   params = model.fit(X_new_train, Y_train, [8, 64, 32, 16, 8, 4, 2, 1])
5. Make predictions using the trained neural network:
   ```code
   Y_pred_train = model.predict(X_new_train, params)

## LICENCE
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Special thanks to the open-source community for providing valuable resources and insights into neural network implementations.

Feel free to contribute, report issues, or suggest improvements. Happy coding!
