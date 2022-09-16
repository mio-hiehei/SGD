import autograd.numpy as anp
import numpy as np
import pandas as pd
from datetime import datetime
from nltk import flatten

##### ============== Stochastic Gradient Descent

# some data in matrix form
nparam = np.random.randint(40, 50)
n = 1000000
X = np.zeros((n, nparam))
for col in range(X.shape[1]):
    if col == 0:
        X[:, col] = 1
    else:
        X[:, col] = np.random.uniform(0, 7, size = n)

beta = np.random.uniform(0, 3, size = nparam)
y = np.matmul(X, beta)



#### Create function
def gd_fun(X,
           y,
           min_stepsize,
           stepsize,
           learning_rate,
           stop_iter,
           sgd,
           sgd_n):

    start = datetime.now()
    iter = 0
    gd_parameters = pd.Series([])

    names_beta = []
    for i in range(X.shape[1]):
        names_beta.append(f"beta{i}")

    parameter_vector = pd.DataFrame(columns=flatten(['iter',
                                             'ssr_intercept',
                                             names_beta,
                                             'stepsize',
                                             'percent']))
    nparam = X.shape[1]

    # 1 pick random values for parameters in matrix form
    randparam = np.random.uniform(0, 5, size=nparam)
    ## 2 take derivative of loss function (sum of squared residuals) with respect to each parameter in it, in Matrix form
    def ssr(y, X, randparam):
        return((y - np.dot(X, randparam)) ** 2)


    while stepsize > min_stepsize and iter <= stop_iter:

        # 3 plug the parameter values into the derivatives
        ## 3.1 do a stochastic selection of data points
        if(sgd == True):
            rand = np.random.randint(0, X.shape[1], sgd_n)
            ssr_rand = ssr(y = y[rand],
                           X = X[rand, ],
                           randparam = randparam)

            grad = np.gradient(ssr_rand)

        else:
            ssr_norm = ssr(y = y,
                           X = X,
                           randparam = randparam)

            grad = np.gradient(ssr_norm)


        # 4 calculate new step sizes
        stepsize = grad.sum() * learning_rate
        randparam = randparam - stepsize

        gd_parameters = pd.DataFrame(data = flatten([iter,
             grad[0],
             randparam,
             stepsize,
             min_stepsize / stepsize]),
                                     columns =flatten(['iter',
                                             'ssr_intercept',
                                             names_beta,
                                             'stepsize',
                                             'percent']))




        parameter_vector.append(gd_parameters)

        print(gd_parameters)
        print(f"Done: {iter}")
        iter = iter + 1


    return(parameter_vector)


output = gd_fun(X = X,
           y = y,
           min_stepsize = 0.00000000000000000000000000000000000000001,
           stepsize = 0.0000000000000000000000000001,
           learning_rate = 0.000000000000000000000000000001,
           stop_iter = 100000,
           sgd = True,
           sgd_n = 1000)

print(output)
print(beta)
