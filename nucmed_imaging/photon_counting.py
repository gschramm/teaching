# script to show realation of binomial and poisson distribution in photon counting
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.stats import binom, poisson

n_exp    = 100000   # number of experiments to run
chunk_size = np.uint(1e3)
mu       = 7
seed     = 0 

#---------------------------------------------------------------------------------------------------
np.random.seed(seed)

#for n_nuclei in [10,20,100,1000,10000]:    # number of nuclei to start with
for n_nuclei in [10,20,100,1000]:    # number of nuclei to start with
  
  p = mu/n_nuclei  # probability that an event (decay + detection) occurs
  
  
  if (n_exp) < chunk_size:
    r = np.random.rand(n_exp, n_nuclei)
    # number of events in every experiment
    n = (r<=p).sum(1)
  else:
    nchunks = math.ceil(n_exp / chunk_size)
  
    n = []
    for i in range(nchunks):
      print(f'{i+1} / {nchunks}')
      r = np.random.rand(chunk_size, n_nuclei)
      n.append((r<=p).sum(1))
  
    n = np.array(n).flatten()
  
  # histogram the number of detected events per experiement
  h = np.histogram(n, np.arange(n.max() + 2) - 0.5)
  
  # generate x and y for plots to make life easier
  x = (0.5*(h[1][:-1] + h[1][1:])).astype(np.uint) # the bin centers
  y = h[0]
  
  # calculate binom and poisson pmfs
  xx   = np.arange(n_nuclei+1)
  bpmf = binom.pmf(xx, n_nuclei, p)
  ppmf = poisson.pmf(xx, n_nuclei*p)
  
  # print max abs and relative error
  abs_error     = ppmf - bpmf
  rel_error     = abs_error / bpmf
  print(f'max abs error P/B {abs_error.max()}')
  print(f'max rel error P/B {rel_error.max()}')
  
  fig, ax = plt.subplots(1,1, figsize = (7,4))
  ax.bar(x, y/y.sum(), width = 1, color = 'silver')
  ax.plot(xx, bpmf, 'o-', color = 'tab:blue', label = f'Binomial pmf (N={n_nuclei:.1e}, p = {p:.1e})')
  ax.plot(xx, ppmf, '.:', color = 'tab:orange', label = f'Poisson pmf (mu = N*p = {mu})')
  ax.set_xlim(-0.2,19)
  ax.set_ylim(-0.002,0.3)
  ax.set_xlabel('n (number of events)')
  ax.set_title(f'histogram N = {n_nuclei}, p = {p}, {n_exp} exps.')
  ax.grid(ls = ':')
  ax.legend()
  fig.tight_layout()
  fig.savefig(f'mu_{mu}_p_{p:.1e}.png')
  fig.show()
