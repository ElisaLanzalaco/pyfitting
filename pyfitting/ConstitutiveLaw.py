import numpy as np
import matplotlib.pyplot as plt
from inspect import signature

#==============================================================================#
class ConstitutiveLaw(object):
    """ Class meant to select the constitutive model for regression. """

    def __init__(self, x, y, param0,
            nature = "linear",
            analysis = "univariate",
            model = None, avg=None, std=None):

        self.nature = nature
        self.analysis = analysis
        self.nparam = param0.shape[0]

        if analysis == "univariate" and len(y.shape) > 1:
            raise ValueError("The analysis is set as UNIVARIATE, but the data is MULTIVARIATE.")
        if analysis == "multivariate" and len(y.shape) == 1:
            raise ValueError("The analysis is set as MULTIVARIATE, but the data is UNIVARIATE.")

        if nature == "linear":
            if analysis == "univariate":
                funcs = self.LinearFunctionUni
            elif analysis == "multivariate":
                funcs = self.LinearFunctionMulti
            else:
                raise ValueError("The function is linear but its analysis is not understood.")
        elif nature == "nonlinear":
            if model == "yeoh":
                if analysis == "univariate":
                    #funcs = self.YeohUniaxial
                    funcs = self.YeohBiaxial1
                elif analysis == "multivariate":
                    funcs = self.YeohBiaxial2
            elif model == "fung":
                funcs = self.FungOrthotropic
            elif model == "hsgr" or model == "holzapfel-sommer":
                funcs = self.HSGR
            elif model == "mhgo" or model == "holzapfel-mod":
                funcs = self.MHGO
            elif model == "may" or model == "may-newmann":
                funcs = self.MayNewmann
            elif model == "mmay" or model == "may-newmann-mod":
                funcs = self.MMayNewmann
            elif model == "myocardium":
                funcs = self.HOMyocardium
            elif model == "fan_sacks":
                if avg==None or std==None:
                    raise ValueError("Average or Standard deviation not set for structural model.")
                funcs = self.FanSacks
                self.avg = avg
                self.std = std
            else:
                raise ValueError("Nonlinear function, but the analysis is not understood.")

            sig = signature(funcs)
            no_sig = len(sig.parameters)-1
            print("The function takes the following arguments, {}.".format(sig))
            print("Then it needs {} parameters for optimization.".format(no_sig))
            if no_sig != self.nparam:
                raise ValueError("The number of parameters given does not match"+\
                                 " with the number of parameters needed in the function")

        self.funcs = funcs

        return

#------------------------------------------------------------------------------#
    """Linear functions for the univariate and multivariate cases"""

    def LinearFunctionUni(self, x, b0, b1, b2=None):

        if len(x.shape) == 1:
            y = b0 + b1*x
        else:
            y = b0 + b1*x[:,0] + b2*x[:,1]

        return y

    def LinearFunctionMulti(self, x, b10, b11, b12, b20, b21, b22):

        y = np.zeros((x.shape[0],2),dtype=np.float64)
        y[:,0] = b10 + b11*x[:,0] + b12*x[:,1]
        y[:,1] = b20 + b21*x[:,0] + b22*x[:,1]

        return y

#------------------------------------------------------------------------------#
    """Nonlinear functions for biaxial tests"""

    def YeohBiaxial1(self, x, c10, c20, c30):

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        I1 = 2.0*x**2 + 1.0/(x**2)
        stress = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x**2 - x**(-4))

        return stress

#------------------------------------------------------------------------------#
    def YeohBiaxial2(self, x, c10, c20, c30):

        x1 = x[:,0]
        x2 = x[:,1]

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        I1 = x1**2 + x2**2 + 1.0/(x1*x2)
        stress[:,0] = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x1**2 - (x1*x2)**(-2))
        stress[:,1] = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x2**2 - (x1*x2)**(-2))

        return stress

#------------------------------------------------------------------------------#
    def FungOrthotropic(self, x, c, b11, b22, b33, b12, b13, b23):
        """Fung function for orthotropic stress"""

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        e1 = 0.5*(x1**2-1.0)
        e2 = 0.5*(x2**2-1.0)
        e3 = 0.5*(x3**2-1.0)

        Q = b11*e1**2 + b22*e2**2 + b33*e3**2 + \
            2.0*b12*e1*e2 + 2.0*b13*e1*e3 + 2.0*b23*e2*e3 #+ \
            #b44*e4**2 + b55*e5**2 + b66*e6**2

        sum1 = (b11*e1 + b12*e2 + b13*e3)*x1**2
        sum2 = (b12*e1 + b22*e2 + b23*e3)*x2**2
        sum3 = (b13*e1 + b23*e2 + b33*e3)*x3**2

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = c*np.exp(Q)*(sum1 - sum3)
        stress[:,1] = c*np.exp(Q)*(sum2 - sum3)

        return stress

#--------------------------------------------------------------------------
    def HSGR(self, x, c0, c1, c2, phi_f):
        """compute the stress from Holzapfel-Sommer-Gasser-Regitnig model."""

        #phi_f = 46.0*np.pi/180.0
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        I1 = x1**2 + x2**2 + x3**2
        I4f = (x1*np.cos(phi_f))**2 + (x2*np.sin(phi_f))**2

        Q_f = c1*(I1 - 3.0)**2 + c2*(I4f - 1.0)**2

        sum_iso1 = (I1-3.0)*(x1**2 - x3**2)
        sum_f1 = (I4f-1.0)*(x1*np.cos(phi_f))**2

        sum_iso2 = (I1-3.0)*(x2**2 - x3**2)
        sum_f2 = (I4f-1.0)*(x2*np.sin(phi_f))**2

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = 4.0*c0*np.exp(Q_f)*(c1*sum_iso1 + c2*sum_f1)
        stress[:,1] = 4.0*c0*np.exp(Q_f)*(c1*sum_iso2 + c2*sum_f2)

        return stress

#--------------------------------------------------------------------------
    def MayNewmann(self, x, c0, c1, c2, phi_f):
        """compute the stress from May-Newman and Yin model."""

        #phi_f = 60.0*np.pi/180.0
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        I1 = x1**2 + x2**2 + x3**2
        I4f = (x1*np.cos(phi_f))**2 + (x2*np.sin(phi_f))**2

        Q_f = c1*(I1-3.0)**2 + c2*(np.sqrt(I4f) - 1.0)**4

        sum_iso1 = (I1-3.0)*(x1**2 - x3**2)
        sum_f1 = ((np.sqrt(I4f)-1.0)**3/np.sqrt(I4f))*(x1*np.cos(phi_f))**2

        sum_iso2 = (I1-3.0)*(x2**2 - x3**2)
        sum_f2 = ((np.sqrt(I4f)-1.0)**3/np.sqrt(I4f))*(x2*np.sin(phi_f))**2

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = 4.0*c0*np.exp(Q_f)*( c1*sum_iso1 + c2*sum_f1 )
        stress[:,1] = 4.0*c0*np.exp(Q_f)*( c1*sum_iso2 + c2*sum_f2 )

        return stress
#------------------------------------------------------------------------------#
    def HOMyocardium(self, x, a, b, a_f, b_f, a_s, b_s, a_fs, b_fs, phi_f):
        """Holzapfel-Ogden function for passive mechanical behavior of myocardium"""

        phi_s = phi_f - 0.5*np.pi

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        I1 = x1**2 + x2**2 + x3**2
        I4f = (x1*np.cos(phi_f))**2 + (x2*np.sin(phi_f))**2
        I4s = (x1*np.cos(phi_s))**2 + (x2*np.sin(phi_s))**2
        I8 = x1*x1*np.cos(phi_f)*np.cos(phi_s) + x2*x2*np.sin(phi_f)*np.sin(phi_s)

        Q_iso = b*(I1 - 3.0)
        Q_f = b_f*(I4f - 1.0)**2
        Q_s = b_s*(I4s - 1.0)**2
        Q_fs = b_fs*I8**2

        sum_iso1 = x1**2 - x3**2
        sum_f1 = (x1*np.cos(phi_f))**2
        sum_s1 = (x1*np.cos(phi_s))**2
        sum_fs1 = x1*x1*np.cos(phi_f)*np.cos(phi_s)

        sum_iso2 = x2**2 - x3**2
        sum_f2 = (x2*np.sin(phi_f))**2
        sum_s2 = (x2*np.sin(phi_s))**2
        sum_fs2 = x2*x2*np.sin(phi_f)*np.sin(phi_s)

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = a*np.exp(Q_iso)*sum_iso1 \
                    + 2.0*a_f*(I4f-1.0)*np.exp(Q_f)*sum_f1 \
                    + 2.0*a_s*(I4s-1.0)*np.exp(Q_s)*sum_s1 \
                    + 2.0*a_fs*I8*np.exp(Q_fs)*sum_fs1
        stress[:,1] = a*np.exp(Q_iso)*sum_iso2 \
                    + 2.0*a_f*(I4f-1.0)*np.exp(Q_f)*sum_f2 \
                    + 2.0*a_s*(I4s-1.0)*np.exp(Q_s)*sum_s2 \
                    + 2.0*a_fs*I8*np.exp(Q_fs)*sum_fs2

        return stress

#------------------------------------------------------------------------------#
    def MHGO(self, x, c10, a_f, b_f, phi_f):
        """Modified Holzapfel-Gasser-Ogden function for mechanical behavior
        W = p*(J-1) + 0.5*(a/b)*(exp(b*(I1-3))-1) + 0.5*(af/bf)*(exp(bf*(I4-1)**2)-1)
        S = p*J*C^-1 + a*exp(b*(I1-3))*I + 2*af*(I4-1)*exp(bf*(I4-1)**2)*a0*a0

        W = p*(J-1) + c10*(I1-3) + 0.5*(af/bf)*(exp(bf*(I4-1)**2)-1)
        S = p*J*C^-1 + 2*c10*I + 2*af*(I4-1)*exp(bf*(I4-1)**2)*a0*a0
        """

        a_s = a_f
        b_s = b_f
        phi_s = -phi_f # - 0.5*np.pi

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        #I1 = x1**2 + x2**2 + x3**2
        I4f = (x1*np.cos(phi_f))**2 + (x2*np.sin(phi_f))**2
        I4s = (x1*np.cos(phi_s))**2 + (x2*np.sin(phi_s))**2

        #Q_iso = b*(I1 - 3.0)
        Q_f = b_f*(I4f - 1.0)**2
        Q_s = b_s*(I4s - 1.0)**2
        #Q_fs = b_fs*I8**2

        sum_iso1 = x1**2 - x3**2
        sum_f1 = (x1*np.cos(phi_f))**2
        sum_s1 = (x1*np.cos(phi_s))**2
        #sum_fs1 = x1*x1*np.cos(phi_f)*np.cos(phi_s)

        sum_iso2 = x2**2 - x3**2
        sum_f2 = (x2*np.sin(phi_f))**2
        sum_s2 = (x2*np.sin(phi_s))**2
        #sum_fs2 = x2*x2*np.sin(phi_f)*np.sin(phi_s)

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = 2.0*c10*sum_iso1 \
                    + 2.0*a_f*(I4f-1.0)*np.exp(Q_f)*sum_f1 \
                    + 2.0*a_s*(I4s-1.0)*np.exp(Q_s)*sum_s1 
        #            + 2.0*a_fs*I8*np.exp(Q_fs)*sum_fs1
        stress[:,1] = 2.0*c10*sum_iso2 \
                    + 2.0*a_f*(I4f-1.0)*np.exp(Q_f)*sum_f2 \
                    + 2.0*a_s*(I4s-1.0)*np.exp(Q_s)*sum_s2 
        #            + 2.0*a_fs*I8*np.exp(Q_fs)*sum_fs2

        return stress

#--------------------------------------------------------------------------
    def MMayNewmann(self, x, c10, af, bf, phi_f):
        """Modified May-Newmann and Yin function for mechanical behavior
        W = p*(J-1) + 0.5*(a/b)*(exp(b*(I1-3))-1) 
                    + 0.5*(af/bf)*(exp(bf*(sqrt(I4)-1)**4)-1)
        S = p*J*C^-1 + a*exp(b*(I1-3))*I 
                     + 4*af*((sqrt(I4)-1)**3/sqrt(I4))*exp(bf*(sqrt(I4)-1)**4)*a0*a0

        W = p*(J-1) + c10*(I1-3) + 0.5*(af/bf)*(exp(bf*(sqrt(I4)-1)**4)-1)
        S = p*J*C^-1 + 2*c10*I 
                     + 4*af*((sqrt(I4)-1)**3/sqrt(I4))*exp(bf*(sqrt(I4)-1)**4)*a0*a0
        """

        phi_s = -phi_f
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        #I1 = x1**2 + x2**2 + x3**2
        I4f = (x1*np.cos(phi_f))**2 + (x2*np.sin(phi_f))**2
        I4s = (x1*np.cos(phi_s))**2 + (x2*np.sin(phi_s))**2

        #Q_iso = b*(I1-3.0)
        Q_f = bf*(np.sqrt(I4f) - 1.0)**4
        Q_s = bf*(np.sqrt(I4s) - 1.0)**4

        sum_iso1 = x1**2 - x3**2
        sum_f1 = ((np.sqrt(I4f)-1.0)**3/np.sqrt(I4f))*(x1*np.cos(phi_f))**2
        sum_s1 = ((np.sqrt(I4s)-1.0)**3/np.sqrt(I4s))*(x1*np.cos(phi_s))**2

        sum_iso2 = x2**2 - x3**2
        sum_f2 = ((np.sqrt(I4f)-1.0)**3/np.sqrt(I4f))*(x2*np.sin(phi_f))**2
        sum_s2 = ((np.sqrt(I4s)-1.0)**3/np.sqrt(I4s))*(x2*np.sin(phi_s))**2

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = 2.0*c10*sum_iso1 \
                    + 4.0*af*np.exp(Q_f)*sum_f1 \
                    + 4.0*af*np.exp(Q_s)*sum_s1
        stress[:,1] = 2.0*c10*sum_iso2 \
                    + 4.0*af*np.exp(Q_f)*sum_f2 \
                    + 4.0*af*np.exp(Q_s)*sum_s2

        return stress

#--------------------------------------------------------------------------
    def FanSacks(self, x, mu, c0, c1, E_ub, d):
        """Simplified Fan and Sacks function for mechanical behavior
        W = p*(J-1) + 0.5*mu*(I1-3) 
                    + \int Gamma(theta)*Psi(E_ens)*dtheta
        S = p*J*C^-1 + mu*I 
                    + \int Gamma(theta)*S_ens(E_ens)*a0*a0*dtheta

        """

        avg = self.avg
        std = self.std
        from scipy.special import erf
        # integration points and intervals
        xquad=np.array([-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846], dtype=np.float64)
        wquad=np.array([0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269], dtype=np.float64)
        ninter = 20
        nquad = xquad.shape[0]

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        stress_m = np.zeros((x.shape[0],2), dtype=np.float64)
        stress_m[:,0] = mu*(x1**2 - x3**2)
        stress_m[:,1] = mu*(x2**2 - x3**2)

        stress_ens = np.zeros((x.shape[0],2), dtype=np.float64)
        stress_f = 0.0
        interval = np.pi/float(ninter)
        for i in range(ninter):
            interval0 = np.pi*float(i)/float(ninter)-0.5*np.pi
            stress_fq = 0.0
            for j in range(nquad):
                theta = 0.5*interval*(xquad[j]+1.0) + interval0
                I4 = (x1*np.cos(theta))**2 + (x2*np.sin(theta))**2
                E_ens = 0.5*(I4-1.0)
                if theta-avg>0.5*np.pi:
                    gamma = d*np.exp(-0.5*((theta-avg-np.pi)/std)**2)/\
                        (erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
                        (1.0-d)/np.pi
                elif theta-avg<-0.5*np.pi:
                    gamma = d*np.exp(-0.5*((theta-avg+np.pi)/std)**2)/\
                        (erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
                        (1.0-d)/np.pi
                else:
                    gamma = d*np.exp(-0.5*((theta-avg)/std)**2)/\
                        (erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
                        (1.0-d)/np.pi
                S_ens = np.zeros((E_ens.shape[0]), dtype=np.float64)
                for k in range(E_ens.shape[0]):
                    if E_ens[k] <= E_ub:
                        S_ens[k] = c0*(np.exp(c1*E_ens[k])-1.0)
                    else:
                        S_ens[k] = c0*(np.exp(c1*E_ub)-1.0) + c0*c1*np.exp(c1*E_ub)*(E_ens[k]-E_ub)
                stress_ens[:,0] = gamma*S_ens*(x1*np.cos(theta))**2
                stress_ens[:,1] = gamma*S_ens*(x2*np.sin(theta))**2
                stress_fq += wquad[j]*stress_ens
            stress_f += interval*stress_fq

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = stress_m[:,0] + stress_f[:,0]
        stress[:,1] = stress_m[:,1] + stress_f[:,1]

        return stress

#------------------------------------------------------------------------------#
    """Nonlinear functions for uniaxial tests"""

    def YeohUniaxial(self, x, c10, c20, c30):

        I1 = x**2 + 2./x

        stress = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x**2 - 1.0/x)

        return stress

#------------------------------------------------------------------------------#
    def ChiSquare(self, param, x, y, sigma2_inv):
        """mrqcof: used by mrqmin to evaluate the linearized fitting matrix
        alpha, and the vector beta, and calculate chisq"""

        # x (IN):          set of datapoints. [ndata]
        # y (IN):          set of datapoints. [ndata]
        # sigma2 (IN):     estimated variance.
        # param (INOUT):   coefficients of a nonlinear function. [ma]
        # chisq (OUT):     chi square, the functional to minimize
        # funcs (IN):      the nonlinear function to fit, funcs(x,a,yfit,dyda,ma)


        if len(y.shape) == 1:
            f_model = self.funcs(x,*param)

            error = y - f_model

            chi_square = np.dot(error,error)*sigma2_inv

        else:
            f_model = self.funcs(x,*param)

            error = y - f_model

            chi_square = np.einsum("ij,jk,ik",error,sigma2_inv,error)

        return chi_square

#------------------------------------------------------------------------------#
    def ChiSquareDifferential(self, param, param_bool, x, y, sigma2_inv):
        """mrqcof: used by mrqmin to evaluate the linearized fitting matrix
        alpha, and the vector beta, and calculate chisq"""

        # x (IN):          set of datapoints. [ndata]
        # y (IN):          set of datapoints. [ndata]
        # sigma2 (IN):     estimated variance.
        # param (INOUT):   coefficients of a nonlinear function. [ma]
        # param_bool (IN): boolean array to select the coeff to be fitted. [ma]
        # alpha (OUT):     curvature matrix (hessian). [ma,ma]
        # beta (OUT):      vector (first derivatives). [ma]
        # chisq (OUT):     chi square, the functional to minimize
        # funcs (IN):      the nonlinear function to fit, funcs(x,a,yfit,dyda,ma)

        if len(y.shape) == 1:
            df = np.zeros((x.shape[0],param.shape[0]),dtype=np.float64)
            err = np.zeros((x.shape[0],param.shape[0]),dtype=np.float64)
            for i in range(param.shape[0]):
                for j in range(x.shape[0]):
                    df[j,i],err[j,i] = self.RiddersMethodUni(i,j,param,x)

            f_model = self.funcs(x,*param)

            error = y - f_model

            hessian0 = np.einsum("ij,ik->jk",df,df)
            hessian0 = sigma2_inv*hessian0
            hessian = hessian0[param_bool,:][:,param_bool]

            gradient0 = np.einsum("i,ik->k",error,df)
            gradient0 = sigma2_inv*gradient0
            gradient = gradient0[param_bool]

        else:
            df = np.zeros((x.shape[0],y.shape[1],param.shape[0]),dtype=np.float64)
            err = np.zeros((x.shape[0],y.shape[1],param.shape[0]),dtype=np.float64)
            for i in range(param.shape[0]):
                for k in range(y.shape[1]):
                    for j in range(x.shape[0]):
                        df[j,k,i],err[j,k,i] = self.RiddersMethodMulti(i,j,k,param,x)

            f_model = self.funcs(x,*param)

            error = y - f_model

            hessian0 = np.einsum("ijl,jk->ikl",df,sigma2_inv)
            hessian0 = np.einsum("ijk,ijl->kl",hessian0,df)
            hessian = hessian0[param_bool,:][:,param_bool]

            gradient0 = np.einsum("ij,jk->ik",error,sigma2_inv)
            gradient0 = np.einsum("ij,ijk->k",gradient0,df)
            gradient = gradient0[param_bool]

        return hessian, gradient

#------------------------------------------------------------------------------#
    def GetEstimatedVariance(self, x, y, param):

        nparam = self.nparam   # number of coefficients in the function
        ndata = y.shape[0]     # number of coefficients in the function

        if len(y.shape) == 1:
            f_model = self.funcs(x,*param)
            error = y - f_model

            sigma2 = np.dot(error,error)
            sigma2 /= (ndata-nparam)

        else:
            f_model = self.funcs(x,*param)
            error = y - f_model

            sigma2 = np.einsum("ij,ik->jk",error,error)
            sigma2 /= (ndata-nparam)

        return sigma2

#------------------------------------------------------------------------------#
    """
    Returns the derivative of a function func at a point x by Ridders method of polynomial
    extrapolation. The value h is input as an estimated initial stepsize; it need not be small,
    but rather should be an increment in x over which func changes substantially. An estimate
    of the error in the derivative is returned as err .
    Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
    NTAB. Return when error is SAFE worse than the best so far.
    """
    def RiddersMethodMulti(self, idx, jdx, kdx, cte, xdata):
        h=1.0e-3
        BIG=1.0e30
        NTAB=10
        CON=1.4
        CON2=CON*CON
        SAFE=2.0
        a = np.zeros((NTAB,NTAB),dtype=np.float64)
        hh = np.zeros(cte.shape,dtype=np.float64)
        hh[idx] = cte[idx]*h
        cte_low = cte - hh
        cte_up = cte + hh
        a[0,0] = (self.funcs(xdata,*cte_up)[jdx,kdx]-\
                self.funcs(xdata,*cte_low)[jdx,kdx])/(2.0*hh[idx])
        err=BIG
        # Successive columns in the Neville tableau will go to smaller stepsizes
        # and higher orders of extrapolation.
        for i in range(1,NTAB):
            hh=hh/CON
            cte_low = cte - hh
            cte_up = cte + hh
            # Try new, smaller stepsize.
            a[0,i] = (self.funcs(xdata,*cte_up)[jdx,kdx]-\
                    self.funcs(xdata,*cte_low)[jdx,kdx])/(2.0*hh[idx])
            fac=CON2
            # Compute extrapolations of various orders, requiring no new function evaluations.
            for j in range(1,i+1):
                a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.0)
                fac=CON2*fac
                errt=max(abs(a[j,i]-a[j-1,i]),abs(a[j,i]-a[j-1,i-1]))
                #The error strategy is to compare each new extrapolation to one order lower, both at
                #the present stepsize and the previous one.
                #If error is decreased, save the improved answer.
                if (errt<=err):
                    err=errt
                    dfridr=a[j,i]
            #If higher order is worse by a significant factor SAFE, then quit early.
            if (abs(a[i,i]-a[i-1,i-1])>=SAFE*err):
                #print('Early quit in df_ridders function')
                return dfridr,err
        return dfridr,err

    def RiddersMethodUni(self, idx, jdx, cte, xdata):
        h=1.0e-3
        BIG=1.0e30
        NTAB=10
        CON=1.4
        CON2=CON*CON
        SAFE=2.0
        a = np.zeros((NTAB,NTAB),dtype=np.float64)
        hh = np.zeros(cte.shape,dtype=np.float64)
        hh[idx] = h
        cte_low = cte - hh
        cte_up = cte + hh
        a[0,0] = (self.funcs(xdata,*cte_up)[jdx]-\
                self.funcs(xdata,*cte_low)[jdx])/(2.0*hh[idx])
        err=BIG
        # Successive columns in the Neville tableau will go to smaller stepsizes
        # and higher orders of extrapolation.
        for i in range(1,NTAB):
            hh=hh/CON
            cte_low = cte - hh
            cte_up = cte + hh
            # Try new, smaller stepsize.
            a[0,i] = (self.funcs(xdata,*cte_up)[jdx]-\
                    self.funcs(xdata,*cte_low)[jdx])/(2.0*hh[idx])
            fac=CON2
            # Compute extrapolations of various orders, requiring no new function evaluations.
            for j in range(1,i+1):
                a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.0)
                fac=CON2*fac
                errt=max(abs(a[j,i]-a[j-1,i]),abs(a[j,i]-a[j-1,i-1]))
                #The error strategy is to compare each new extrapolation to one order lower, both at
                #the present stepsize and the previous one.
                #If error is decreased, save the improved answer.
                if (errt<=err):
                    err=errt
                    dfridr=a[j,i]
            #If higher order is worse by a significant factor SAFE, then quit early.
            if (abs(a[i,i]-a[i-1,i-1])>=SAFE*err):
                #print('Early quit in df_ridders function')
                return dfridr,err
        return dfridr,err

#------------------------------------------------------------------------------#
    def MakePlots(self, x, y, params):
        """This is a function to make a general plot of the fitting"""

        fig, ax = plt.subplots(constrained_layout=True)

        # plot curve and data
        ax.plot(x, self.funcs(x, *params), 'b-', label='fitting')
        ax.plot(x, y, 'o', label='data')

        x_max = 1.01*np.max(x)
        y_max = 1.10*np.max(y)
        # make the graphic nicer
        ax.set_xlabel('stretch',fontsize=14)
        ax.set_ylabel('stress',fontsize=14)
        ax.set_ylim(bottom=0,top=y_max)
        ax.set_xlim(left=1.0,right=x_max)
        for label in (ax.get_xticklabels()+ax.get_yticklabels()):
            label.set_fontsize(14)
        #open the box and add legend
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left',fontsize=14)

        plt.show
        #file output
        FIGURENAME = 'fitting.pdf'
        plt.savefig(FIGURENAME)
        #close graphical tools
        plt.close('all')

        if len(y.shape) != 1:
            from mpl_toolkits.mplot3d import Axes3D

            X = np.linspace(1.0,x_max,7)
            Y = np.linspace(1.0,x_max,7)
            X, Y = np.meshgrid(X,Y)
            Z1 = np.zeros((X.shape[0],X.shape[1]),dtype=np.float64)
            Z2 = np.zeros((X.shape[0],X.shape[1]),dtype=np.float64)
            for i in range(X.shape[0]):
                XY_data = np.column_stack((X[i,:],Y[i,:]))
                Z1[i,:] = self.funcs(XY_data,*params)[:,0]
                Z2[i,:] = self.funcs(XY_data,*params)[:,1]

            # first plot
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')

            ax.plot_wireframe(X, Y, Z1)
            ax.plot(x[:,0],x[:,1],y[:,0], 'b-')

            ax.set_xlabel('stretch 1',fontsize=11)
            ax.set_ylabel('stretch 2',fontsize=11)
            ax.set_zlabel('stress',fontsize=11)

            # second plot
            ax = fig.add_subplot(1, 2, 2, projection='3d')

            ax.plot_wireframe(X, Y, Z2)
            ax.plot(x[:,0],x[:,1],y[:,1], 'r-')

            ax.set_xlabel('stretch 1',fontsize=11)
            ax.set_ylabel('stretch 2',fontsize=11)
            ax.set_zlabel('stress',fontsize=11)

            plt.show

            FIGURENAME = 'fitting_3d.pdf'
            plt.savefig(FIGURENAME)

            plt.close('all')


