from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_samples = X.shape[0]

    # scores = np.dot(X, W) # (N, C)
    # scores -= np.max(scores) # numerical stability

    loss = 0
    for sample_idx in range(num_samples):
        scores = X[sample_idx].dot(W)  # (C,)
        scores -= np.max(scores)

        exp_sum = np.sum(np.exp(scores))

        target_vector = np.zeros_like(scores)
        target_vector[y[sample_idx]] = 1
        for y_class in range(num_classes):
            probability = np.exp(scores[y_class]) / exp_sum            
            probability_error = probability - target_vector[y_class]

            dW[:,y_class] += probability_error * X[sample_idx]            

            if y_class == y[sample_idx]:
                loss -= np.log(probability)



    # compute the mean of sample losses and gradients
    loss /= num_samples
    dW /= num_samples

    # add regularization loss and gradients
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_samples = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)   # (N, C)
    scores -= np.max(scores)   # for numerical stability

    exp_sums = np.sum(np.exp(scores), axis=1)  # (N,)

    # compute the scores for the target class
    target_class_scores = scores[np.arange(num_samples), y]   # (N,)
    loss -= np.sum(np.log(np.exp(target_class_scores) / exp_sums))
    
    probabilities = np.exp(scores) / exp_sums[:, np.newaxis]  # (N, C)
    errors = probabilities
    errors[np.arange(num_samples), y] -= 1   

    dW += X.T.dot(errors)

    dW /= num_samples
    loss /= num_samples
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
