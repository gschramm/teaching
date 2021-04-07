# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom
from pyparallelproj.models import pet_fwd_model, pet_back_model

from matplotlib.colors import LogNorm
import numpy as np
import argparse

plt.rcParams['image.cmap'] = 'Greys'

def osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
         fwhm = 0, verbose = False, xstart = None, 
         callback = None, subset_callback = None,
         callback_kwargs = None, subset_callback_kwargs = None, vmax = None):

  img_shape  = tuple(proj.img_dim)

  # calculate the sensitivity images for each subset
  sens_img  = np.zeros((proj.nsubsets,) + img_shape, dtype = np.float32)
 
  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss        = proj.subset_slices[i]
    # generate a subset sinogram full of ones
    ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)
    sens_img[i,...] = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
  
  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, (em_sino.sum()/sens_img.sum()), dtype = np.float32)
  else:
    recon = xstart.copy()

  plt.ion()
  fig, ax = plt.subplots(2,5, figsize = (15,6))

  ax[0,3].imshow(em_sino.squeeze().T, aspect = 'auto', vmin = 0, vmax = em_sino.max()) 

  for axx in ax.ravel():
    axx.set_axis_off()

  # run OSEM iterations
  for it in range(niter):
    for i in range(proj.nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')

      # get the slice for the current subset
      ss        = proj.subset_slices[i]

      exp_sino = pet_fwd_model(recon, proj, attn_sino[ss], sens_sino[ss], i, 
                               fwhm = fwhm) + contam_sino[ss]
      ratio  = em_sino[ss] / exp_sino

      if (it == 0) and (i == 0):
        p14 = ax[1,4].imshow(recon.squeeze(), vmin = 0, vmax = vmax) 
        p00 = ax[0,0].imshow(recon.squeeze(), vmin = 0, vmax = vmax) 
      else:
        p14.set_data(recon.squeeze()) 
        p00.set_data(recon.squeeze()) 

      back_ratio = pet_back_model(ratio, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)


      recon *= (back_ratio / sens_img[i,...]) 

      if (it == 0) and (i == 0):
        p13 = ax[1,3].imshow(exp_sino.squeeze().T, aspect = 'auto', vmin = 0, vmax = em_sino.max()) 
        p02 = ax[0,2].imshow(ratio.squeeze().T, aspect = 'auto', vmin = 0.8, vmax = 1.2) 
        p01 = ax[0,1].imshow(back_ratio.squeeze(), 
                             vmin = sens_img[i,...].min(), vmax = sens_img[i,...].max())
        p11 = ax[1,1].imshow(sens_img[i,...].squeeze(),
                             vmin = sens_img[i,...].min(), vmax = sens_img[i,...].max())
        p10 = ax[1,0].imshow(recon.squeeze(), vmin = 0, vmax = vmax) 
        fig.tight_layout()
      else:
        p13.set_data(exp_sino.squeeze().T)
        p02.set_data(ratio.squeeze().T)
        p01.set_data(back_ratio.squeeze())
        p11.set_data(sens_img[i,...].squeeze())
        p10.set_data(recon.squeeze()) 


      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon


#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 4,   type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus         = args.ngpus
counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
  n   = 200
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = ellipse_phantom(n = n, c = 3)
  for i2 in range(n2):
    img[:,:,i2] = tmp
elif phantom == 'brain2d':
  n   = 128
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = brain2d_phantom(n = n)
  for i2 in range(n2):
    img[:,:,i2] = tmp

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01 * voxsize[0]

# generate nonTOF sinogram parameters and the nonTOF projector for attenuation projection
sino_params = ppp.PETSinogramParameters(scanner, rtrim = 146)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)

attn_sino = np.exp(-proj.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = np.ones(sino_params.shape, dtype = np.float32)

# forward project the image
img_fwd= ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, 0, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(0*img[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[1].set_title('recon')
ib = ax[2].imshow(img[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                  cmap = plt.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()

#-----------------------------------------------------------------------------------------------
# callback functions to calculate likelihood and show recon updates

def update_img(x):
  ir.set_data(x[...,n2//2])
  ib.set_data(x[...,n2//2] - img[...,n2//2])
  plt.pause(1e-6)

def calc_cost(x):
  cost = 0

  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    cost += (exp - em_sino[ss]*np.log(exp)).sum()

  return cost

def _cb(x, **kwargs):
  """ This function is called by the iterative recon algorithm after every iteration 
      where x is the current reconstructed image
  """
  it = kwargs.get('iteration',0)
  update_img(x)
  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)

#-----------------------------------------------------------------------------------------------
# run the actual reconstruction using OSEM

# initialize the subsets for the projector
proj.init_subsets(nsubsets)

cost_osem = np.zeros(niter)
cbk       = {'cost':cost_osem}

init_recon = 0*img + img.mean()

recon_osem = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter, 
                  fwhm = fwhm, verbose = True, xstart = init_recon, vmax = 1.2*img.max(),
                  callback = _cb, callback_kwargs = cbk)

#-----------------------------------------------------------------------------------------------
# plot the cost function

init_cost = calc_cost(init_recon) 
ref_cost  = cost_osem[-1]
fig2, ax2 = plt.subplots(1,1, figsize = (6,4))
ax2.semilogy(np.arange(1,niter+1), (cost_osem - ref_cost) / (init_cost - ref_cost), '.-')
ax2.set_xlabel('iteration')
ax2.set_ylabel('relative cost')
fig2.tight_layout()
fig2.show()
