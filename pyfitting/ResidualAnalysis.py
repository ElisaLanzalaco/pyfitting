import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#==============================================================================#
class ResidualAnalysis(object):
    """ Class meant to measure the quality of the regression. """
    def __init__(self, x, y, params, param_bool, model):

        deter = self.DeterminationCoefficient(x, y, params, model.funcs)
        self.determination = deter

        sigma2 = model.GetEstimatedVariance(x, y, params)
        if len(y.shape) == 1:
            sigma2_inv = 1.0/sigma2
        else:
            sigma2_inv = np.linalg.inv(sigma2)

        hessian, gradient = model.ChiSquareDifferential(params, param_bool, x, y, sigma2_inv)
        covariance = np.linalg.inv(hessian)

        corr, det = self.CorrelationAssessment(covariance)
        self.correlation = corr.copy()
        self.correlation_det = det

        if len(y.shape) == 1:
            error = self.ResidualAnalysisUni(x, y, params, model.funcs)
        else:
            error = self.ResidualAnalysisMulti(x, y, params, model.funcs)

        self.error = error

        return

#------------------------------------------------------------------------------#
    def CorrelationAssessment(self, covariance):

        #covariance = covariance[param_bool,:][:,param_bool]

        #correlation = np.zeros((mfit,mfit),dtype=np.float64)
        correlation = np.zeros(covariance.shape,dtype=np.float64)
        for i in range(covariance.shape[0]):
            for j in range(covariance.shape[1]):
                correlation[i,j] = covariance[i,j]/\
                           np.sqrt(covariance[i,i]*covariance[j,j])

        determinant = np.linalg.det(correlation)

        return correlation, determinant

#------------------------------------------------------------------------------#
    def DeterminationCoefficient(self, x, y, params, funcs):

        from sklearn.metrics import r2_score
        y_pred = funcs(x, *params)
        determination = r2_score(y, y_pred, multioutput='raw_values')

        return determination

#------------------------------------------------------------------------------#
    def ResidualAnalysisUni(self, x, y, params, funcs):

        error = y - funcs(x, *params)

        fig, ax = plt.subplots(1,3, figsize=plt.figaspect(0.3))

        # histogram for error
        ax[0].hist(error, bins=6)
        ax[0].set_xlabel('error',fontsize=11)
        ax[0].set_ylabel('quantity',fontsize=11)

        # scatter stretch against error
        ax[1].scatter(x, error)
        ax[1].set_xlabel('stretch',fontsize=11)
        ax[1].set_ylabel('error',fontsize=11)

        # QQplot for error
        sm.graphics.qqplot(error, line='s', ax=ax[2])

        fig.tight_layout()
        plt.show
        #file output
        FIGURENAME = 'error.pdf'
        plt.savefig(FIGURENAME)
        #close graphical tools
        plt.close('all')

        return error

#------------------------------------------------------------------------------#
    def ResidualAnalysisMulti(self, x, y, params, funcs):

        error = y - funcs(x, *params)

        fig, ax = plt.subplots(2,3, figsize=plt.figaspect(0.5))

        # histogram for error 1
        ax[0,0].hist(error[:,0], bins=6)
        ax[0,0].set_xlabel('error 1',fontsize=11)
        ax[0,0].set_ylabel('quantity',fontsize=11)

        # histogram for error 1
        ax[1,0].hist(error[:,1], bins=6)
        ax[1,0].set_xlabel('error 2',fontsize=11)
        ax[1,0].set_ylabel('quantity',fontsize=11)

        # scatter stretch against error
        ax[0,1].scatter(x[:,0], error[:,0])
        ax[0,1].scatter(x[:,1], error[:,1])
        ax[0,1].set_xlabel('stretch',fontsize=11)
        ax[0,1].set_ylabel('error',fontsize=11)

        # plot error 1 against error 2
        ax[1,1].plot([min(error[:,0]),max(error[:,0])],
                     [min(error[:,0]),max(error[:,0])],
                     color='red')
        ax[1,1].scatter(error[:,0], error[:,1])
        ax[1,1].set_xlabel('error 1',fontsize=11)
        ax[1,1].set_ylabel('error 2',fontsize=11)

        # QQplot for error 1
        sm.graphics.qqplot(error[:,0], line='s', ax=ax[0,2])

        # QQplot for error 2
        sm.graphics.qqplot(error[:,1], line='s', ax=ax[1,2])

        fig.tight_layout()
        plt.show
        #file output
        FIGURENAME = 'error.pdf'
        plt.savefig(FIGURENAME)
        #close graphical tools
        plt.close('all')

        return error


