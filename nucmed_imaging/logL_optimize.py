# demo scirpt that compares L-BFGS-B, MLEM and PDHG for optimizing
# the negative Poisson logL with fwd model A@x + b

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy.optimize import minimize, Bounds

def f(x, A, y, b):
  ybar = (A @ x) + b

  return (ybar - y*np.log(ybar)).sum()

def fdual(y, m):
  if y.max() <= 1:
    r = (-m + m*np.log(m/(1-y))).sum()
  else:
    r = np.inf

  return r

#---------------------------------------------------------------------------------------------
def fprime(x, A, y, b):
  ybar = (A @ x) + b

  return A.transpose() @ (1 - y/ybar)

#---------------------------------------------------------------------------------------------
def hessian(x, A, y, b):
  ybar = (A @ x) + b

  n = x.shape[0]
  H = np.zeros((n,n))

  for i in range(n):
    for j in range(n):
      H[i,j] = (A[:,i]*A[:,j]*y/(ybar**2)).sum()

  return H

#---------------------------------------------------------------------------------------------
def newton(x0,A,y,b, niter = 10):

  x = x0.copy()

  xn = [[x.copy(),f(x,A,y,b)]]

  for i in range(niter):
    x = np.clip(x - np.linalg.inv(hessian(x,A,y,b)) @ fprime(x, A, y, b), 0, None)
    xn.append([x.copy(),f(x,A,y,b)])

  return xn

#---------------------------------------------------------------------------------------------
def mlem(x0, A, y, b, niter = 10):

  x = x0.copy()
  s = A.transpose() @ np.ones(A.shape[0]) 

  xm = [[x.copy(),f(x,A,y,b)]]

  for i in range(niter):
    ybar = (A @ x) + b

    x *= (A.transpose() @ (y/ybar)) / s
    xm.append([x.copy(),f(x,A,y,b)])
  
  return xm

#---------------------------------------------------------------------------------------------
def PDHG(x0, A, m, b, niter = 15, rho = 1., gamma = 1., precond = False):

  if precond:
    S = gamma*rho/(A @ np.ones(A.shape[1]))
    T = rho/(gamma*A.transpose() @ np.ones(A.shape[0]))
  else:
    norm_A = np.sqrt(np.linalg.eig(A.transpose()@A)[0].max())
    S = gamma*rho/norm_A
    T = rho/(gamma*norm_A)

  x = x0.copy()
  y = np.zeros(A.shape[0])

  xp = [[x.copy(), f(x,A,m,b), y.copy(), -fdual(y,m) + (b*y).sum()]]

  for i in range(niter):
    x_plus = np.clip(x - T*(A.transpose() @ y), 0, None)
    
    y_plus = y + S*((A @ (2*x_plus - x)) + b)
    
    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S*m))
   
    # update variables
    x = x_plus.copy()
    y = y_plus.copy()

    xp.append([x.copy(), f(x,A,m,b), y.copy(), -fdual(y,m) + (b*y).sum()])

  return xp

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------


# input parameters
niter = 100
sens  = 3.
noise = True
# additive contaminations in fwd model A@x + b
b     = 1

xdata = np.array([5.,3.])

# forward operator
A = sens*np.array([[6.,0.1],[0.1,3.],[2.,2.]])

#---------------------------------------------------------------------------------------------

np.random.seed(1)

# operator norm (largest eigenvalue)
norm_A = np.sqrt(np.linalg.eig(A.transpose()@A)[0].max())

y = (A @ xdata) + b
print(y)

if noise:
  y = np.random.poisson(y)
print(y)

# choose initial recon
x0 = (A.transpose() @ y) / (norm_A**2)
x0[0] *= 1.5

# l-bfgs-b
xl = [[x0,f(x0,A,y,b)]]
cbl = lambda x: xl.append([x,f(x, A, y, b)])
lb_res = minimize(f, x0, method = 'L-BFGS-B', jac = fprime, args = (A,y,b), callback = cbl,
                  bounds = Bounds(0,10*xdata.max()), 
                  options = {'ftol':1e-15, 'gtol':1e-15, 'maxiter':niter})

# MLEM
xm = mlem(x0, A, y, b, niter = niter)

# full Newton method using hessian
xn = newton(x0, A, y, b, niter = niter)

# PDHG - gamma should be approx 1 / mean of solution
xp  = PDHG(x0, A, y, b, gamma = 1./xdata.max(), niter = niter, precond = True)
xp2 = PDHG(x0, A, y, b, gamma = 0.3/xdata.max(), niter = niter, precond = True)
xp3 = PDHG(x0, A, y, b, gamma = 3./xdata.max(), niter = niter, precond = True)

#----------------------------------------------------------------------------------------------
# show convergence of cost functions
fl  = np.array([x[1] for x in xl])
fm  = np.array([x[1] for x in xm])
fp  = np.array([x[1] for x in xp])
fp2 = np.array([x[1] for x in xp2])
fp3 = np.array([x[1] for x in xp3])
fn  = np.array([x[1] for x in xn])

fmin  = min(fl.min(),fm.min(),fp.min(),fp2.min(),fp3.min(),fn.min())

fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].semilogy(np.arange(len(xl)), fl - fmin, '.-', label = 'L-BFGS-B')
ax[0].semilogy(np.arange(len(xm)), fm - fmin, '.-', label = 'MLEM')
ax[0].semilogy(np.arange(len(xp)), fp - fmin, '.-', label = 'PDHG - gam: 1 / ||x||')
ax[0].semilogy(np.arange(len(xp2)),fp2 - fmin, '.-', label = 'PDHG - gam: 0.3 / ||x||')
ax[0].semilogy(np.arange(len(xp3)), fp3 - fmin, '.-', label = 'PDHG - gam: 3 / ||x||')
ax[0].semilogy(np.arange(len(xn)), fn - fmin, '.-', label = 'Newton w. full Hessian')
ax[0].grid(':')
ax[0].legend()
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('-logL + logL(x*)')

#----------------------------------------------------------------------------------------------
# show contour plot
n1 = 100
n2 = 120
F = np.zeros((n1,n2))
xx = np.linspace(min(xdata[0],x0[0])/100, 1.5*max(xdata[0],x0[0]), n1)
yy = np.linspace(min(xdata[1],x0[1])/100, 1.5*max(xdata[1],x0[1]), n2)

for i, xt0 in enumerate(xx):
  for k, xt1 in enumerate(yy):
    F[i,k] = f(np.array([xt0,xt1]), A, y, b)

ax[1].contour(yy, xx, np.log(F - fmin + 1), levels = 50, colors = 'k', linewidths = 0.5)
ax[1].grid(':')
ax[1].plot([x[0][1] for x in xl], [x[0][0] for x in xl], '.-')
ax[1].plot([x[0][1] for x in xm], [x[0][0] for x in xm], '.-')
ax[1].plot([x[0][1] for x in xp], [x[0][0] for x in xp], '.-')
ax[1].plot([x[0][1] for x in xp2], [x[0][0] for x in xp2], '.-')
ax[1].plot([x[0][1] for x in xp3], [x[0][0] for x in xp3], '.-')
ax[1].plot([x[0][1] for x in xn], [x[0][0] for x in xn], '.-')
ax[1].set_ylabel('x0')
ax[1].set_xlabel('x1')

# plot primal dual Gaps for PDHGs
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax[2].semilogy([abs(xp[x][-1] - xp[x][-3]) for x in range(1,niter+1)], '.-', color = cols[2])
ax[2].semilogy([abs(xp2[x][-1] - xp2[x][-3]) for x in range(1,niter+1)], '.-', color = cols[3])
ax[2].semilogy([abs(xp3[x][-1] - xp3[x][-3]) for x in range(1,niter+1)], '.-', color = cols[4])
ax[2].set_xlabel('iteration')
ax[2].set_ylabel('primal dual gap')
ax[2].grid(':')

fig.tight_layout()
fig.show()
