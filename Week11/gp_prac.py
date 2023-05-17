import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

def K(X, X_star, length_scale):
    """
    Args:
        X (nparray): (n, d)
        X_star (nparray): (m, d)
        length_scale (float): 
    """
    dist = sp.distance.cdist(X, X_star)
    return np.exp(-dist**2/(2*length_scale**2))

def sample_prior(X, length_scale, N):
    """
    Args:
        X (nparray): (n, d)
        length_scale (float): 
        N (int): 
    """
    n = X.shape[0]
    k = K(X, X, length_scale)
    samples = np.random.multivariate_normal([0]*n, k, N)
    return samples

def predictive_mean(X, y, X_star, length_scale, sigma):
    """
    Args:
        X (nparray): (n, d)
        y (nparray): (n, 1)
        X_star (nparray): (m, d)
        length_scale (float): 
        sigma (float)
    """
    n = X.shape[0]
    K_xstar_x = K(X_star, X, length_scale)
    K_x_x = K(X, X, length_scale)

    return \
    np.matmul(K_xstar_x, np.matmul(np.linalg.inv(K_x_x+sigma**2*np.eye(n)), y))

def predictive_cov(X, y, X_star, length_scale, sigma):
    """
    Args:
        X (nparray): (n, d)
        y (nparray): (n, 1)
        X_star (nparray): (m, d)
        length_scale (float): 
        sigma (float):
    """

    n = X.shape[0]
    K_xstar_xstar = K(X_star, X_star, length_scale)
    K_xstar_x = K(X_star, X, length_scale)
    K_x_x = K(X, X, length_scale)

    return K_xstar_xstar - np.matmul(K_xstar_x,
            np.matmul(np.linalg.inv(K_x_x + sigma**2*np.eye(n)), K_xstar_x.T))

if __name__ == '__main__':
    # Load test data
    test_inputs = np.loadtxt('test_inputs').reshape((-1,1))
    train_inputs = np.loadtxt('train_inputs').reshape((-1,1))
    train_outputs = np.loadtxt('train_outputs').reshape((-1,1))

    # Test Q1 answer
    k = K(test_inputs, test_inputs, 1)
    plt.imshow(k)
    plt.savefig('kernel.png')
    plt.close()

    # Test Q2 and Q3 answer
    samples = sample_prior(test_inputs, 1, 5)
    [plt.plot(test_inputs, samples[i, :]) for i in range(5)]
    plt.savefig('prior.png')
    plt.close()

    # Tests Q4 and Q5 answer
    mean = predictive_mean(train_inputs, train_outputs, test_inputs, 
            1, 1).reshape((-1,))
    cov = predictive_cov(train_inputs, train_outputs, test_inputs, 
            1, 1)
    var = np.diag(cov)

    plt.plot(train_inputs, train_outputs, 'k*')
    plt.plot(test_inputs, mean, 'r')
    plt.fill_between(test_inputs.reshape((-1,)), 
            mean + np.sqrt(var), mean - np.sqrt(var))
    plt.savefig('pred.png')
    plt.close()
    
    # Test Q6
    samples = np.random.multivariate_normal(mean, cov, 5)
    plt.plot(train_inputs, train_outputs, 'k*')
    [plt.plot(test_inputs, samples[i, :]) for i in range(5)]
    plt.savefig('posterior.png')
    plt.close()

