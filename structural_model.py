# python3
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

rad2grad = 180.0/np.pi

#------------------------------------------------------------------------------#
xquad=np.array([-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846], dtype=np.float64)
wquad=np.array([0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269], dtype=np.float64)

ninter = 20

# fibre distribution
d = 0.85
avg = 0.5*np.pi
std = 0.155*np.pi

# mechanical properties
mu = 125.0e-3
c0 = 400.0e-3
c1 = 10.0
E_ub = 0.15

#------------------------------------------------------------------------------#
def odf(theta, d, avg, std):

#    gamma = d*np.exp(-0.5*((theta-avg)/std)**2)/\
#            (special.erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
#            (1.0-d)/np.pi

    if theta-avg>0.5*np.pi:
        gamma = d*np.exp(-0.5*((theta-avg-np.pi)/std)**2)/\
            (special.erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
            (1.0-d)/np.pi
    elif theta-avg<-0.5*np.pi:
        gamma = d*np.exp(-0.5*((theta-avg+np.pi)/std)**2)/\
            (special.erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
            (1.0-d)/np.pi
    else:
        gamma = d*np.exp(-0.5*((theta-avg)/std)**2)/\
            (special.erf(np.pi/(2.0*std*np.sqrt(2.0)))*std*np.sqrt(2.0*np.pi)) + \
            (1.0-d)/np.pi

    return gamma

#------------------------------------------------------------------------------#
def fibre_stress(theta, F):

    detF_2d = F[0,0]*F[1,1] - F[0,1]*F[1,0]
    N_0 = np.array([np.cos(theta), np.sin(theta), 0.0])
    N_t = np.dot(F,N_0)
    outerNN = np.outer(N_0,N_0)
    inv4 = np.inner(N_t,N_t)
    E_ens = 0.5*(inv4-1.0)

    gamma = odf(theta, d, avg, std)
    if E_ens <= E_ub:
        S_ens = c0*(np.exp(c1*E_ens)-1.0)
    else:
        S_ens = c0*(np.exp(c1*E_ub)-1.0) + c0*c1*np.exp(c1*E_ub)*(E_ens-E_ub)

    stress = gamma*S_ens*outerNN

    beta = np.arctan(N_t[1]/N_t[0])
    gamma_t = gamma*inv4/detF_2d

    return stress, beta, gamma_t

#------------------------------------------------------------------------------#
def cauchy_stress(F):

    # isotropic part
    I = np.eye(3)
    C = np.dot(np.transpose(F),F)
    Cinv = np.linalg.inv(C)
    stress_m = mu*(I-Cinv*C[2,2])

    # fibre family part
    # define theta at gaussian points for each interval, -pi/2 to pi/2
    nquad = xquad.shape[0]
    beta = np.zeros((ninter*nquad), dtype=np.float64)
    gamma_t = np.zeros((ninter*nquad), dtype=np.float64)
    stress_f = 0.0
    interval = np.pi/float(ninter)
    for i in range(ninter):
        interval0 = np.pi*float(i)/float(ninter)-0.5*np.pi
        stress_fq = 0.0
        for j in range(nquad):
            theta = 0.5*interval*(xquad[j]+1.0) + interval0
            stress_ens, beta[i*nquad+j], gamma_t[i*nquad+j] = fibre_stress(theta,F)
            stress_fq += wquad[j]*stress_ens
        stress_f += interval*stress_fq            # LOOK HERE, MAYBE THERE IS A MISTAKE

    stress = stress_m + stress_f

    strain = 0.5*(C-I)

    return stress, strain, beta, gamma_t

#------------------------------------------------------------------------------#
def fibre_distribution(theta, gamma):

    # index for min and max gamma
    id_valley = np.argmin(gamma)
    id_peak = np.argmax(gamma)
    id_min = np.argmin(theta)

#    print("\nangle at Gamma-peak = {}".format(rad2grad*theta[id_peak]))
#    print("angle at Gamma-valley = {}".format(rad2grad*theta[id_valley]))

    # make theta continuous
    theta0 = np.copy(theta)
    theta0[:id_min] = theta[:id_min] - np.pi

    # phase theta and gamma to have a symmetrical distribution
    center_sum = 0.0; var_sum = 0.0; weight_sum = 0.0
    area_bars = np.zeros((theta.shape[0]), dtype=np.float64)
    area_p = np.zeros((50), dtype=np.float64)
    half_theta = int(theta.shape[0]/2)
    delta_theta = theta0 - theta0[id_peak]
    cos2_oi = gamma[id_peak]*np.cos(delta_theta[id_peak])**2
    for i in range(1,half_theta):
        lb = id_peak-i
        ub = id_peak+i

        if lb == -1:
            area_bars[-1] = 0.5*(gamma[0]+gamma[-1])*(theta0[0]-theta0[-1] + np.pi)
            center_sum += 0.5*(gamma[-1] + gamma[0]) \
                * (theta0[-1] - theta0[id_peak] - np.pi)
            var_sum += 0.5*(gamma[-1] + gamma[0]) \
                * (theta0[-1] - theta0[id_peak] - np.pi)**2
            weight_sum += 0.5*(gamma[-1] + gamma[0])
        elif lb < -1:
            area_bars[lb] = 0.5*(gamma[lb+1]+gamma[lb])*(theta0[lb+1]-theta0[lb])
            center_sum += 0.5*(gamma[lb] + gamma[lb+1]) \
                * (theta0[lb] - theta0[id_peak] - np.pi)
            var_sum += 0.5*(gamma[lb] + gamma[lb+1]) \
                * (theta0[lb] - theta0[id_peak] - np.pi)**2
            weight_sum += 0.5*(gamma[lb] + gamma[lb+1])
        else:
            area_bars[lb] = 0.5*(gamma[lb+1]+gamma[lb])*(theta0[lb+1]-theta0[lb])
            center_sum += 0.5*(gamma[lb] + gamma[lb+1]) \
                * (theta0[lb] - theta0[id_peak])
            var_sum += 0.5*(gamma[lb] + gamma[lb+1]) \
                * (theta0[lb] - theta0[id_peak])**2
            weight_sum += 0.5*(gamma[lb] + gamma[lb+1])

        if ub > theta.shape[0]:
            area_bars[ub-100] = 0.5*(gamma[ub-100]+gamma[ub-101]) \
                * (theta0[ub-100]-theta0[ub-101])
            center_sum += 0.5*(gamma[ub-100] + gamma[ub-101]) \
                * (theta0[ub-100] - theta0[id_peak] + np.pi)
            var_sum += 0.5*(gamma[ub-100] + gamma[ub-101]) \
                * (theta0[ub-100] - theta0[id_peak] + np.pi)**2
            weight_sum += 0.5*(gamma[ub-100] + gamma[ub-101])
        elif ub < theta.shape[0]:
            area_bars[ub] = 0.5*(gamma[ub]+gamma[ub-1])*(theta0[ub]-theta0[ub-1])
            center_sum += 0.5*(gamma[ub] + gamma[ub-1]) \
                * (theta0[ub] - theta0[id_peak])
            var_sum += 0.5*(gamma[ub] + gamma[ub-1]) \
                * (theta0[ub] - theta0[id_peak])**2
            weight_sum += 0.5*(gamma[ub] + gamma[ub-1])
        else:
            area_bars[0] = 0.5*(gamma[0]+gamma[-1])*(theta0[0]-theta0[-1] + np.pi)
            center_sum += 0.5*(gamma[0] + gamma[-1]) \
                * (theta0[0] - theta0[id_peak] + np.pi)
            var_sum += 0.5*(gamma[0] + gamma[-1]) \
                * (theta0[0] - theta0[id_peak] + np.pi)**2
            weight_sum += 0.5*(gamma[0] + gamma[-1])

        # compute orientation index
        if ub>99:
            area_p[i] = area_p[i-1] + area_bars[ub-100] + area_bars[lb]
            cos2_oi += gamma[lb]*np.cos(delta_theta[lb])**2 + gamma[ub-100]*np.cos(delta_theta[ub-100])**2
        else:
            area_p[i] = area_p[i-1] + area_bars[ub] + area_bars[lb]
            cos2_oi += gamma[lb]*np.cos(delta_theta[lb])**2 + gamma[ub]*np.cos(delta_theta[ub])**2

    center = center_sum/weight_sum
    std = np.sqrt(var_sum/weight_sum - center**2)
#    print("Centroid = {}".format(rad2grad*(center+theta0[id_peak])))  #center+theta0[id_peak]
#    print("Standard Deviation = {}".format(rad2grad*std))

    area_total = area_p[-1]
    area_p_unit = area_p/area_total
    id_area50p = np.argmin(np.abs(area_p_unit-0.5))
    theta_p_i = theta0[id_peak - id_area50p]
    if id_peak + id_area50p<100:
        theta_p_f = theta0[id_peak + id_area50p]
    else:
        theta_p_f = theta0[id_peak + id_area50p - 100]

    if theta_p_f<theta_p_i: theta_p_f += np.pi
    oi_p = theta_p_f - theta_p_i
    noi_p = (0.5*np.pi-oi_p)/(0.5*np.pi)
#    print("Orientation index (peak): OI={}, NOI={}".format(rad2grad*oi_p, noi_p))
    oi2 = oi_p

    oi_cos = cos2_oi/weight_sum
#    print("Orientation index (cos) = {}".format(oi_cos))

    return oi2, oi_cos, center, std

#------------------------------------------------------------------------------#
# kinematics
stretchx = np.linspace(1.0, 1.118, num=10)
stretchy = np.linspace(1.0, 1.026, num=10)
distortion = np.linspace(0.0, 0.0, num=10)

stress = np.zeros((stretchx.shape[0],3), dtype=np.float64)
strain = np.zeros((stretchx.shape[0],3), dtype=np.float64)
orindx = np.zeros((stretchx.shape[0]), dtype=np.float64)
oricos = np.zeros((stretchx.shape[0]), dtype=np.float64)
for i in range(stretchx.shape[0]):
    stretch1 = stretchx[i]
    stretch2 = stretchy[i]
    kappa = distortion[i]
    stretch3 = 1.0/(stretch1*stretch2 - kappa**2)
    F = np.array([[stretch1, kappa, 0.0],
                  [kappa, stretch2, 0.0],
                  [0.0, 0.0, stretch3]], dtype=np.float64)

    #------------------------------------------------------------------------------#
    # COMPUTATION OF STRESS
    stress_t, strain_t, beta, gamma_t = cauchy_stress(F)

    stress[i,0] = stress_t[0,0]
    stress[i,1] = stress_t[1,1]
    stress[i,2] = stress_t[0,1]

    strain[i,0] = strain_t[0,0]
    strain[i,1] = strain_t[1,1]
    strain[i,2] = strain_t[0,1]

    if i==0:
        beta_0 = beta
        gamma_0 = gamma_t
    #------------------------------------------------------------------------------#
    # identify intensity minima to isolate symmetric segment
    orindx[i], oricos[i], center0, std0 = fibre_distribution(beta, gamma_t)

#------------------------------------------------------------------------------#
exp_data = np.loadtxt('pcuu_pcl_9_1.dat')

#------------------------------------------------------------------------------#
# graphic of fibre distribution
plt.rcParams.update({'font.size':10})
fig, ax = plt.subplots(2,2, figsize=plt.figaspect(1.0))

ax[0,0].plot(np.sqrt(2.0*strain[:,0]+1.0),(np.pi-2.0*orindx)/np.pi, 'k')
ax[0,0].set_xlabel("stretch 1")
ax[0,0].set_ylabel("normalized orientation index")
ax[0,0].set_ylim(0.4,0.9)

ax[0,1].plot(rad2grad*beta_0,gamma_0, 'k:', label='reference')
ax[0,1].plot(rad2grad*beta,gamma_t, 'k', label='spatial')
ax[0,1].set_xlabel(r"$\theta$, $\beta$")
ax[0,1].set_ylabel(r"$\Gamma(\theta)$, $\Gamma_t(\beta)$")
ax[0,1].set_xlim(-90,90)
ax[0,1].set_ylim(0,1)
ax[0,1].legend()

ax[1,0].plot(np.sqrt(2.0*strain[:,0]+1.0),oricos, 'k')
ax[1,0].set_xlabel("stretch 1")
ax[1,0].set_ylabel("orientation index (cos)")
ax[1,0].set_ylim(0.4,0.9)

#ax[1,1].plot(np.sqrt(2.0*strain[:,0]+1.0),1000.0*stress[:,0], label='CD')
#ax[1,1].plot(np.sqrt(2.0*strain[:,1]+1.0),1000.0*stress[:,1], label='RD')
ax[1,1].plot(strain[:,0], 1000.0*stress[:,0], 'b', label='$S_{11}$')
ax[1,1].plot(strain[:,1], 1000.0*stress[:,1], 'r', label='$S_{22}$')
ax[1,1].plot(strain[:,2], 1000.0*stress[:,2], 'g', label='$S_{12}$')
#ax[1,1].plot(strain[:,2],1000.0*stress[:,2])
ax[1,1].plot(exp_data[:,5], 1e3*exp_data[:,9], 'b:')
ax[1,1].plot(exp_data[:,8], 1e3*exp_data[:,12], 'r:')
ax[1,1].plot(exp_data[:,6], 1e3*exp_data[:,10], 'g:')
ax[1,1].set_xlabel("Green strain")
ax[1,1].set_ylabel("Lagrangian stress [kPa]")
ax[1,1].legend()

fig.tight_layout()
plt.show
FIGURENAME = 'structural.pdf'
plt.savefig(FIGURENAME)
plt.close('all')

