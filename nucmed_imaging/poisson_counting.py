# small demo script to visualize poisson nature of (photon) counting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from scipy.stats import poisson
from time import sleep

n_exp       = 10000     # number of experiments to simulate
n_epochs    = 100       # number of epochs to simulated ("aquisition time")
p_emission  = 0.96      # probability that an emission happens in one epoch

phi_dec     = np.pi/16  # accepctance angle of the detector
seed        = 0         # seed for the random generator

n_animate   = 5         # animated the last experiment and save to animated gif

#-----------------------------------------------------------------------------------------

# expected number of detected photons per experiment
exp_n_det = n_epochs*p_emission*phi_dec/(2*np.pi)

np.random.seed(seed)

n_em  = np.zeros(n_exp, dtype = np.uint)
n_det = np.zeros(n_exp, dtype = np.uint)
f_det = np.zeros(n_exp, dtype = np.float)

for i,seed in enumerate(range(n_exp)): 

  # random number whether emission happened in one epoch
  em = (np.random.rand(n_epochs) >= (1 - p_emission)).astype(np.uint8)
  
  # 2D emission angle of event
  phi_em = 2*np.pi*np.random.rand(n_epochs) - np.pi
  
  event_detected = em * (np.abs(phi_em) <= phi_dec/2)
 
  n_em[i]  = em.sum()
  n_det[i] = event_detected.sum()    # total number of detected photons
  f_det[i] = n_det[i] / n_em[i]      # ratio of detected to emitted photons
  
  print(f'{i+1} emitted events {n_em[i]}  detected events: {n_det[i]}  ratio {f_det[i]:.2e}')


  #-----------------------------------------------------------------------------------------------------
  # animate the last experiment and save to animated gif

  if i < n_animate:
    from celluloid import Camera
    
    cols = ['tab:blue','tab:orange']
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    cam  = Camera(fig)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_axis_off()
    
    # add the detector
    rect = patches.Rectangle((1, -np.tan(phi_dec/2)), 0.4, 2*np.tan(phi_dec/2), linewidth=1, 
                             edgecolor='none', facecolor='black')
    
    n_det_cum = np.cumsum(event_detected)
    n_em_cum  = np.cumsum(em)
    
    for k in range(n_epochs):
      ax.add_patch(rect)
      if em[k] == 1:
        line  = ax.plot([0,np.cos(phi_em[k])],[0,np.sin(phi_em[k])], color = cols[event_detected[k]])
      plt.legend(line, [f'experiment {i+1} - epoch {k+1} - emitted events {n_em_cum[k]} - detected events {n_det_cum[k]}'])
      cam.snap()
    
    animation = cam.animate()
    animation.save(f'photon_counting_{i+1}.gif', writer = 'imagemagick')

#-----------------------------------------------------------------------------------------------------

# histogram the number of detected events per experiement
h = np.histogram(n_det, np.arange(n_det.max() + 2) - 0.5)

# generate x and y for plots to make life easier
x = 0.5*(h[1][:-1] + h[1][1:]) # the bin centers
y = h[0]

fig, ax = plt.subplots(1,1, figsize = (7,4))
ax.bar(x, y/y.sum(), width = 1)
ax.plot(x, poisson.pmf(x, exp_n_det), '.-', color = 'tab:orange', 
                       label = r'$P(n,\lambda = {0:.1f})$'.format(exp_n_det) + r'$= \frac{e^{-\lambda} \lambda^n}{n!}$')
ax.set_xlabel('n')
ax.set_title(f'histogram of number of detected events in {n_exp} experiments')
ax.grid(ls = ':')
ax.legend(loc = 'upper right')
fig.tight_layout()
fig.show()
