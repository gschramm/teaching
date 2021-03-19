# small demo script to visualize poisson nature of (photon) counting

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

n_exp       = 50000     # number of experiments to simulate
n_epochs    = 100       # number of epochs to simulated ("aquisition time")
p_emission  = 0.96      # probability that an emission happens in one epoch

phi_dec     = np.pi/16  # accepctance angle of the detector

start_seed  = 0

#-----------------------------------------------------------------------------------------

# expected number of detected photons per experiment
exp_n_det = n_epochs*p_emission*phi_dec/(2*np.pi)

np.random.seed(0)

n_em  = np.zeros(n_exp, dtype = np.uint)
n_det = np.zeros(n_exp, dtype = np.uint)
f_det = np.zeros(n_exp, dtype = np.float)

for i,seed in enumerate(range(n_exp)): 
  # random number whether emission happened in one epoch
  em = (np.random.rand(n_epochs) >= (1 - p_emission)).astype(np.uint8)
  
  # 2D emission angle of event
  phi_em = 2*np.pi*np.random.rand(n_epochs)
  
  event_detected = em * (phi_em <= phi_dec)
 
  n_em[i]  = em.sum()
  n_det[i] = event_detected.sum()    # total number of detected photons
  f_det[i] = n_det[i] / n_em[i]      # ratio of detected to emitted photons
  
  print(f'{i} emitted events {n_em[i]}  detected events: {n_det[i]}  ratio {f_det[i]:.2e}')


# histogram the number of detected events per experiement
h = np.histogram(n_det, np.arange(n_det.max() + 2) - 0.5)

# generate x and y for plots to make life easier
x = 0.5*(h[1][:-1] + h[1][1:]) # the bin centers
y = h[0]

fig, ax = plt.subplots(1,1, figsize = (7,4))
ax.bar(x, y/y.sum(), width = 1)
ax.plot(x, poisson.pmf(x, exp_n_det), '.-', color = 'tab:orange')
ax.grid(ls = ':')
fig.tight_layout()
fig.show()
