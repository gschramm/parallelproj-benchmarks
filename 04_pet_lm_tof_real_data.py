"""OSEM reconstruction example using simulated brainweb data"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from array_api_compat import to_device
from pathlib import Path
from time import time
import h5py
import scanners
import tof
import parallelproj

parser = argparse.ArgumentParser()
parser.add_argument('--lm_file', type=str, default='data/LIST0000.BLF')
parser.add_argument('--corr_file', type=str, default='data/corrections.h5')
parser.add_argument('--num_events', type=int, default=40000000)
parser.add_argument('--num_iterations', type=int, default=6)
parser.add_argument('--num_subsets', type=int, default=34)
parser.add_argument('--mode',
                    default='GPU',
                    choices=['GPU', 'GPU-torch', 'CPU', 'CPU-torch', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=32)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
parser.add_argument('--presort', action='store_true')
parser.add_argument('--post_sm_fwhm', type=float, default=4.)
parser.add_argument('--symmetry_axis', type=int, default=2, choices=[0, 1, 2])
args = parser.parse_args()

presort = args.presort
post_sm_fwhm = args.post_sm_fwhm

if args.mode == 'GPU':
    if not parallelproj.cuda_present:
        raise ValueError('CUDA not present')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import array_api_compat.cupy as xp
    dev = 'cuda'
elif args.mode == 'GPU-torch':
    if not parallelproj.cuda_present:
        raise ValueError('CUDA not present')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import array_api_compat.torch as xp
    dev = 'cuda'
elif args.mode == 'hybrid':
    if not parallelproj.cuda_present:
        raise ValueError('CUDA not present')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import array_api_compat.numpy as xp
    dev = 'cpu'
elif args.mode == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import array_api_compat.numpy as xp
    dev = 'cpu'
elif args.mode == 'CPU-torch':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import array_api_compat.torch as xp
    dev = 'cpu'
else:
    raise ValueError

np.random.seed(0)

data_str = 'nema_tof_listmode'
if presort:
    data_str += '_presorted'

# image properties
voxel_size = xp.asarray([2.78, 2.78, 2.78], dtype=xp.float32, device=dev)
num_trans = int(215 * 2.78 / voxel_size[0])
num_ax = 71

# scanner properties
symmetry_axis = args.symmetry_axis
fwhm_mm_recon = 4.5
tof_parameters = tof.ge_discovery_mi_tof_parameters

# reconstruction parameters
num_iterations = args.num_iterations
num_subsets = args.num_subsets
threadsperblock = args.threadsperblock
num_events = args.num_events

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'{data_str}__mode_{args.mode}__tpb_{threadsperblock}__numevents_{num_events}__axis_{symmetry_axis}.json'

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#--- setup the scanner geometry ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

scanner = scanners.GEDiscoveryMI(xp,
                                 dev,
                                 num_rings=36,
                                 symmetry_axis=symmetry_axis)

image_shape = 3 * [num_trans]
image_shape[symmetry_axis] = num_ax
image_shape = tuple(image_shape)
image_origin = (-(xp.asarray(image_shape, dtype=xp.float32, device=dev) / 2) +
                0.5) * voxel_size
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#--- calcuate the sensivity image (non-TOF backprojction of mult. corrections) ----------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

if Path(args.corr_file).exists():
    with h5py.File(args.corr_file, 'r') as data:
        all_multiplicative_factors = xp.asarray(data['all_xtals/atten'][:] *
                                                data['all_xtals/sens'][:],
                                                device=dev)
        all_xtals = xp.asarray(data['all_xtals/xtal_ids'][:][:, [1, 0, 3, 2]],
                               device=dev)
else:
    raise ValueError(f'file {args.corr_file} does not exist')

xstart_all = scanner.get_lor_endpoints(all_xtals[:, 0], all_xtals[:, 1])
xend_all = scanner.get_lor_endpoints(all_xtals[:, 2], all_xtals[:, 3])

# calculate the sensitivity image - approximation using non-TOF backprojection
t0 = time()
sens_image = parallelproj.joseph3d_back(xstart_all,
                                        xend_all,
                                        image_shape,
                                        image_origin,
                                        voxel_size,
                                        all_multiplicative_factors,
                                        threadsperblock=threadsperblock)

t1 = time()
print(f'time to calculate non-tof adjoint ones {(t1-t0):.2F}s')

# replace zeros (outside FOV) with small value to avoid NaNs
sens_image = xp.where(sens_image < 1e-7, 1e-7, sens_image)

del xstart_all
del xend_all
del all_multiplicative_factors
del all_xtals

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#--- load listmode data and corrections ------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

if Path(args.lm_file).exists():
    with h5py.File(args.lm_file, 'r') as data:
        events = xp.asarray(data['MiceList/TofCoinc'][:], device=dev)
else:
    raise ValueError(f'file {args.lm_file} does not exist')

if num_events is None:
    num_events = events.shape[0]

with h5py.File(args.corr_file, 'r') as data:
    multiplicative_correction_list = xp.asarray(
        data['correction_lists/sens'][:] * data['correction_lists/atten'][:],
        device=dev)
    contamination_list = xp.asarray(data['correction_lists/contam'][:],
                                    device=dev)

# shuffle events since events come semi sorted
print('shuffling LM data')
num_all_events = events.shape[0]
ie = np.arange(num_all_events)
np.random.shuffle(ie)
ie = xp.asarray(ie, device=dev)
events = xp.take(events, ie, axis=0)
multiplicative_correction_list = xp.take(multiplicative_correction_list, ie)
contamination_list = xp.take(contamination_list, ie)

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed
events[:, -1] = -(events[:, -1] // 13)

## use only part of the events
if num_events is not None:
    multiplicative_correction_list = multiplicative_correction_list[:
                                                                    num_events]
    contamination_list = contamination_list[:num_events] * (num_events /
                                                            num_all_events)
    events = events[:num_events, :]

# calculate LOR start and end points
xstart = scanner.get_lor_endpoints(events[:, 0], events[:, 1])
xend = scanner.get_lor_endpoints(events[:, 2], events[:, 3])
tofbin = xp.astype(events[:, 4], xp.int16)

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#---LM OSEM reconstruction ------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

image = xp.ones(image_shape, dtype=xp.float32, device=dev)
sigma_tof = xp.asarray([tof_parameters.sigma_tof],
                       dtype=xp.float32,
                       device=dev)
tofcenter_offset = xp.asarray([tof_parameters.tofcenter_offset],
                              dtype=xp.float32,
                              device=dev)

# for LM OSEM we have to divide the sensitivity image by the number of subsets
subset_corrected_sens_image = sens_image / num_subsets

iteration_times = []

for i_iter in range(num_iterations):
    print(f'iteration {i_iter}')
    ta = time()
    for i_sub in range(num_subsets):

        # forward model
        image_fwd = multiplicative_correction_list[
            i_sub::num_subsets] * parallelproj.joseph3d_fwd_tof_lm(
                xstart[i_sub::num_subsets, :],
                xend[i_sub::num_subsets, :],
                image,
                image_origin,
                voxel_size,
                tof_parameters.tofbin_width,
                sigma_tof,
                tofcenter_offset,
                tof_parameters.num_sigmas,
                tofbin[i_sub::num_subsets],
                threadsperblock=threadsperblock
            ) + contamination_list[i_sub::num_subsets]

        # backproject "ratio"
        tmp = parallelproj.joseph3d_back_tof_lm(
            xstart[i_sub::num_subsets, :],
            xend[i_sub::num_subsets, :],
            image_shape,
            image_origin,
            voxel_size,
            multiplicative_correction_list[i_sub::num_subsets] / image_fwd,
            tof_parameters.tofbin_width,
            sigma_tof,
            tofcenter_offset,
            tof_parameters.num_sigmas,
            tofbin[i_sub::num_subsets],
            threadsperblock=threadsperblock)

        image *= (tmp / subset_corrected_sens_image)
    tb = time()
    iteration_times.append(tb - ta)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- timing results -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

print(f'time per iteration {iteration_times}s')

res = {
    'iteration time (s)': iteration_times,
    'iteration time mean (s)': np.asarray(iteration_times).mean(),
    'iteration time std (s)': np.asarray(iteration_times).std(),
    'num_iterations': num_iterations,
    'num_subsets': num_subsets,
    'num_events': num_events,
    'presort': presort,
    'mode': args.mode,
    'symmetry axis': symmetry_axis,
    'tpb': threadsperblock,
    'fwhm_mm_recon': fwhm_mm_recon
}

Path(output_dir).mkdir(exist_ok=True, parents=True)
with open(Path(output_dir) / output_file, 'w') as f:
    json.dump(res, f)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---- visualizations -------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# convert recon into numpy array
x_lm = np.asarray(to_device(image, 'cpu'))
x_lm_sm = gaussian_filter(
    x_lm, post_sm_fwhm / (2.35 * np.asarray(to_device(voxel_size, 'cpu'))))

ims = dict(cmap=plt.cm.Greys,
           origin='lower',
           vmin=0,
           vmax=0.35 * num_events / 4e7,
           interpolation='bilinear')

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex='col', sharey='col')
ax[0].imshow(np.take(x_lm, 51, axis=symmetry_axis)[50:-50, 60:-60].T, **ims)
ax[1].imshow(
    np.take(x_lm, num_trans // 2, axis=((symmetry_axis + 2) % 3))[50:-50, :].T,
    **ims)
for axx in ax:
    axx.set_axis_off()
fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4), sharex='col', sharey='col')
ax2[0].imshow(
    np.take(x_lm_sm, 51, axis=symmetry_axis)[50:-50, 60:-60].T, **ims)
ax2[1].imshow(
    np.take(x_lm_sm, num_trans // 2,
            axis=((symmetry_axis + 2) % 3))[50:-50, :].T, **ims)
for axx in ax2:
    axx.set_axis_off()
fig2.tight_layout()
fig2.show()