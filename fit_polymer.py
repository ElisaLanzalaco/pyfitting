# python code for fitting
import sys
from pyfitting import *

#==============================================================================#
log = open("fitting.log", "w")
sys.stdout = log

# read data from file. lagrangian stress
exp_data = np.loadtxt('pig_mv.dat',dtype=float)
#exp_data = np.loadtxt('pig_mv.dat',dtype=float)

ndata = 11
x_data = np.zeros((ndata,2),dtype=np.float64)
y_data = np.zeros((ndata,2),dtype=np.float64)
x_data[:,0] = exp_data[:ndata,0]
y_data[:,0] = exp_data[:ndata,1]
x_data[:,1] = exp_data[ndata:,0]
y_data[:,1] = exp_data[ndata:,1]
#x_data_mean = np.copy(x_data.mean(axis=1))
#y_data_mean = np.copy(y_data.mean(axis=1))

# transform the lagrangian to eulerian (real)
y_data[:,0] *= x_data[:,0]
y_data[:,1] *= x_data[:,1]

# define the parameters of Myocardium to fit
param0 = np.array([20.0,7.5,28.0,0.2,0.92],dtype=np.float64)
low_bound = np.array([0.0,0.0,0.0,0.2,0.92], dtype=np.float64)
up_bound = np.array([25.0,20.0,30.0,0.2,0.92], dtype=np.float64)
param_bool = ~np.isclose(low_bound,up_bound)

model = ConstitutiveLaw(x_data, y_data, param0,
                    nature="nonlinear",
                    analysis="multivariate",
                    model="fan_sacks",
                    avg=0.5*np.pi,
                    std=0.155*np.pi)

print("\n***********************************************************************\n")
# optimization of parameters
param = CurveFit(x_data, y_data, param0, low_bound, up_bound, model)

# print optimization results
print("The parameters are: \n{}".format(param))
model.MakePlots(x_data, y_data, param)

print("\n***********************************************************************\n")
residual = ResidualAnalysis(x_data, y_data, param, param_bool, model)

print("Determination Coefficient R2: {}".format(residual.determination))
if residual.correlation_det < 1.0e-4:
    is_overparameter = True
else:
    is_overparameter = False
print("This is the correlation: \n{}".format(residual.correlation))
print("Over-parameterization: {}, ".format(is_overparameter) + \
      "det(R) = {}".format(residual.correlation_det) )

print("\n***********************************************************************\n")
#variability = VariabilityAssessment()
#variability.BootstrapingMulti(x_data, y_data, param0, param, low_bound,
#        up_bound, residual.error, model, around_func=True)

log.close()

