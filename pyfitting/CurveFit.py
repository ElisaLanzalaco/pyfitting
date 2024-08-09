# python3
import numpy as np
from scipy.optimize import minimize,BFGS,Bounds

#==============================================================================#
#class Regression(object):
#    """ Class meant to perform the regression of the data to a model. """
#    def __init__(self, x, y, param0,
#            model,
#            sigma2 = None,
#            regression = "linear"):
#
#        param = self.CurveFit(x, y, param0, model, sigma2)
#
#        return
#------------------------------------------------------------------------------#
def CurveFit(x, y, param0, low_bound, up_bound, model, sigma2=None, bootstrap=False):

    if sigma2 is None:
        compute_variance = True
        ntrial = 2
        if len(y.shape) == 1:
            sigma2 = 1.0
        else:
            sigma2 = np.eye(y.shape[1],dtype=np.float64)
    else:
        compute_variance = False
        ntrial = 1

    bounds = Bounds(low_bound,up_bound)
    print("===========================================================")
    for i in range(ntrial):
        print("Trial number {} to get parameters".format(i))
        if len(y.shape) == 1:
            sigma2_inv = 1.0/sigma2
        else:
            sigma2_inv = np.linalg.inv(sigma2)
        #param, cov, chi_square = self.NonLinearRegression(x, y, param0, 
        #                                                 sigma2, funcs)
        resmin = minimize(model.ChiSquare, param0,
                        method='trust-constr',
                        jac='2-point', #hess=BFGS(),
                        args=(x,y,sigma2_inv),
                        options={'disp': True},bounds=bounds)
        if not bootstrap:
            print(resmin)
        param = resmin.x
        chi_square = resmin.fun

        print("The parameters are: \n{}".format(param))
        print("chi-square is: {}".format(chi_square))

        if compute_variance:
            sigma2 = model.GetEstimatedVariance(x, y, param)
            print("The estimated variance is: \n{}".format(sigma2))
            compute_variance = False

        print(" ")

    return param

