import time
import argparse
import os
from pathlib import Path
import numpy as np
import scanners

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--num_subsets', type=int, default=34)
parser.add_argument('--mode', default='GPU', choices=['GPU', 'CPU', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=32)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy.array_api as xp
dev = 'cpu'

#if args.mode == 'GPU':
#    import cupy as cp
#    xp = cp
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#elif args.mode == 'hybrid':
#    xp = np
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#elif args.mode == 'CPU':
#    xp = np
#    os.environ['CUDA_VISIBLE_DEVICES'] = ''
#else:
#    raise ValueError

import pandas as pd
import parallelproj
#import pyparallelproj.coincidences as coincidences
#import pyparallelproj.subsets as subsets

num_runs = args.num_runs
threadsperblock = args.threadsperblock
num_subsets = args.num_subsets

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'nontofsinogram__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numsubsets_{num_subsets}.csv'

# image properties
num_trans = 215
num_ax = 71
voxel_size = xp.asarray([2.78, 2.78, 2.78], dtype=xp.float32, device = dev)

# scanner properties
num_rings = 36

#---------------------------------------------------------------------
#sinogram_orders = ('PVR', 'PRV', 'VPR', 'VRP', 'RPV', 'RVP')
sinogram_orders = ('PVR', 'PRV')
symmetry_axes = (0, 1, 2)

df = pd.DataFrame()

for ia, symmetry_axis in enumerate(symmetry_axes):
    lor_descriptor = scanners.GEDiscoveryMILORDescriptor(xp, dev, symmetry_axis=2)
    views = xp.arange(0, lor_descriptor.num_views, num_subsets, device=dev)

    # setup a box like test image
    img_shape = [num_trans, num_trans, num_trans]
    img_shape[symmetry_axis] = num_ax
    img_shape = tuple(img_shape)
    n0, n1, n2 = img_shape

    # setup an image containing a square
    img = xp.zeros(img_shape, dtype=xp.float32, device = dev)
    sl = [
        slice(n0 // 4, 3 * n0 // 4, None),
        slice(n1 // 4, 3 * n1 // 4, None),
        slice(n2 // 4, 3 * n2 // 4, None)
    ]

    sl[symmetry_axis] = slice(0, img.shape[symmetry_axis], None)
    sl = tuple(sl)
    img[sl] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(xp.asarray(img.shape, dtype=xp.float32, device = dev) / 2) +
                  0.5) * voxel_size

    for io, sinogram_order in enumerate(sinogram_orders):
        xstart, xend = lor_descriptor.get_lor_coordinates(
            views,
            sinogram_order=scanners.SinogramSpatialAxisOrder[sinogram_order])

        print(sinogram_order)
        print(symmetry_axis, img_shape)

        for ir in range(num_runs + 1):
            # perform a complete fwd projection
            t0 = time.time()
            img_fwd = parallelproj.joseph3d_fwd(xstart,
                                      xend,
                                      img,
                                      img_origin,
                                      voxel_size,
                                      threadsperblock=threadsperblock)
            t1 = time.time()

            # perform a complete backprojection
            back_img = xp.zeros(img.shape, dtype=xp.float32)
            ones = xp.ones(img_fwd.shape, dtype=xp.float32)
            t2 = time.time()
            back_img = parallelproj.joseph3d_back(xstart,
                                       xend,
                                       img_shape,
                                       img_origin,
                                       voxel_size,
                                       ones,
                                       threadsperblock=threadsperblock)
            t3 = time.time()
            if ir > 0:
                tmp = pd.DataFrame(
                    {
                        'sinogram order':
                        sinogram_order,
                        'symmetry axis':
                        symmetry_axis,
                        'run':
                        ir,
                        't forward (s)':
                        t1 - t0,
                        't back (s)':
                        t3 - t2
                    },
                    index=[0])
                df = pd.concat((df, tmp))

#----------------------------------------------------------------------------
# save results
df['data'] = 'nontof_sinogram'
df['mode'] = args.mode
df['num_subsets'] = num_subsets
df['threadsperblock'] = threadsperblock

Path(output_dir).mkdir(exist_ok=True, parents=True)
df.to_csv(os.path.join(output_dir, output_file), index=False)