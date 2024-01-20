\documentclass[12pt]{article}
\usepackage{amsmath}

\usepackage[utf8]{inputenc}
\usepackage{enumitem}

\usepackage{geometry}
\geometry{margin=1in}  % Adjust this value as needed


\usepackage{titling}
\setlength{\droptitle}{-0.6in}  % Adjust this value as needed


 % Adjust this value as needed

\title{Artificial Neural Network (ANN) Implementation from Scratch in Python}
\date{}
\begin{document}

\maketitle
 
\vspace{-2cm} 
\section*{Overview}
This project is a Python implementation of an Artificial Neural Network (ANN) built from scratch using Object-Oriented Programming (OOP) principles. The purpose of this implementation is to gain a deeper understanding of the inner workings of neural networks by constructing the fundamental components from the ground up.

\section*{Features}
\begin{itemize}[label=--]
    \item \textbf{Feedforward Neural Network:} The implemented neural network supports feedforward operations, allowing it to make predictions on input data.
    \item \textbf{Backpropagation:} Backpropagation is used for training the neural network, optimizing the weights to minimize the error between predicted and actual outputs.
    \item \textbf{Activation Functions:} You can choose from various activation functions, including Sigmoid, Softmax, or ReLU, for the hidden layers.
    \item \textbf{Customizable Architecture:} The number of layers, neurons per layer, and activation functions are customizable to suit different tasks and datasets.
    \item \textbf{Gradient Descent:} Batch Gradient Descent (BGD) is employed as the optimization algorithm.
    \item \textbf{Cost function:} Binary Cross Entropy Loss function was used due to its convex nature so that our Gradient Descent optimization algorithm can effectively converge to the minima.
\end{itemize}

\section*{Findings and Research}
\begin{itemize}[label=--]
\item \textbf{Forward Propagation at any layer l:} \[
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
\]

\[
A^{[l]} = g^{[l]}(Z^{[l]})
\]
Where:
\begin{itemize}
    \item \(Z^{[l]}\) is the weighted sum for layer \(l\),
    \item \(W^{[l]}\) is the weight matrix for layer \(l\),
    \item \(A^{[l-1]}\) is the activation from the previous layer,
    \item \(b^{[l]}\) is the bias for layer \(l\),
    \item \(g^{[l]}\) is the activation function for layer \(l\).
\end{itemize}
\item \textbf{Backward Propagation at any layer l:}
\[
dZ^{[l]} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = dA^{[l]} \cdot g'^{[l]}(Z^{[l]})
\]

\[
dW^{[l]} = \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}
\]

\[
db^{[l]} = \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)}
\]

\[
dA^{[l-1]} = \frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = W^{[l]T} \cdot dZ^{[l]}
\]

Where:
\begin{itemize}
    \item \(dZ^{[l]}\) is the gradient of the cost with respect to the weighted sum for layer \(l\),
    \item \(dW^{[l]}\) is the gradient of the cost with respect to the weights for layer \(l\),
    \item \(db^{[l]}\) is the gradient of the cost with respect to the biases for layer \(l\),
    \item \(dA^{[l-1]}\) is the gradient of the cost with respect to the activation from the previous layer,
    \item \(g'^{[l]}\) is the derivative of the activation function for layer \(l\),
    \item \(m\) is the number of examples in the training set.
\end{itemize}
\item \textbf{Comparison of Model performance:} 
\begin{itemize}[label=--]
    \item \textbf{Training Accuracy: }
    \begin{itemize}[label=--]
        \item \textbf{Logistic Regression (sklearn):} 77\%
        \item \textbf{Custom-built ANN:} 65.63\%
    \end{itemize}
    \item \textbf{Testing Accuracy: }
    \begin{itemize}[label=--]
        \item \textbf{Logistic Regression (sklearn):} 78\%
        \item \textbf{Custom-built ANN:} 63\%
    \end{itemize}
    \item There could be various reasons for the observed differences in accuracy:
    \begin{enumerate}
    \item \textbf{Model Complexity:} Logistic regression is a simpler model, suitable for linearly separable problems. If the data is not well-suited for the complexity of a neural network, logistic regression may outperform it.
    
    \item \textbf{Data Size:} The size of the dataset can influence model performance. Smaller datasets may lead to better generalization for simpler models like logistic regression.
    
    \item \textbf{Hyperparameter Tuning:} Absence of hyperparameter tuning such as learning rate, architectue etc. might also be the reason.
    \end{enumerate}
    
\end{itemize}
\end{itemize}

\section*{Dependencies}
\begin{itemize}
    \item Python 3.x
    \item NumPy: \texttt{pip install numpy}
\end{itemize}

\section*{Usage}
\begin{enumerate}
    \item Clone the repository:\\ 
        git clone https://github.com/PriyanshuKSG/WIDS\_final\_submission\_22B2165.git
    
    \item Import the \texttt{ANN\_Priyanshu} class into your Python script or Jupyter Notebook:
    \begin{verbatim}
        from basic_binary_classification import ANN_Priyanshu
    \end{verbatim}
    
    \item Create an instance of the \texttt{ANN\_Priyanshu} class and configure the architecture:
    \begin{verbatim}
        model = ANN_Priyanshu()
    \end{verbatim}
    
    \item Train the neural network on your dataset:
    \begin{verbatim}
        params = model.fit(X_new_train, Y_train, [8, 64, 32, 16, 8, 4, 2, 1])
    \end{verbatim}
    
    \item Make predictions using the trained neural network:
    \begin{verbatim}
        Y_pred_train = model.predict(X_new_train, params)
    \end{verbatim}
\end{enumerate}

\section*{License}
This project is licensed under the MIT License - see the \texttt{LICENSE} file for details.

\section*{Acknowledgments}
Special thanks to the open-source community for providing valuable resources and insights into neural network implementations.

Feel free to contribute, report issues, or suggest improvements. Happy coding!

\end{document}
