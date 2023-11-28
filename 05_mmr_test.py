# script to benchmark performance of projections for mMR geometry
# we use a "small" max ring difference to get a total number of planes
# that is similar to the 837 "span 11" planes
# script added to compare performance against NiftyPET
import time
import argparse
import os
import scanners
import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--num_subsets', type=int, default=1)
parser.add_argument('--mode',
                    default='GPU',
                    choices=['GPU', 'GPU-torch', 'CPU', 'CPU-torch', 'hybrid'])
parser.add_argument('--threadsperblock', type=int, default=32)
parser.add_argument('--output_file', type=int, default=None)
parser.add_argument('--output_dir', default='results')
parser.add_argument('--sinogram_orders',
                    default=['RVP'],
                    nargs='+')
parser.add_argument('--symmetry_axes', default=['2'], nargs='+')

args = parser.parse_args()

if args.mode == 'GPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import array_api_compat.cupy as xp
    dev = 'cuda'
elif args.mode == 'GPU-torch':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import array_api_compat.torch as xp
    dev = 'cuda'
elif args.mode == 'hybrid':
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

import parallelproj

num_runs = args.num_runs
threadsperblock = args.threadsperblock
num_subsets = args.num_subsets

output_dir = args.output_dir
if args.output_file is None:
    output_file = f'mmr_nontof_sinogram__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numsubsets_{num_subsets}.csv'

# image properties
num_trans = 344
num_ax = 127
voxel_size = xp.asarray([2.08, 2.08, 2.03], dtype=xp.float32, device=dev)

sinogram_orders = args.sinogram_orders
symmetry_axes = [int(x) for x in args.symmetry_axes]

#---------------------------------------------------------------------

df = pd.DataFrame()

for ia, symmetry_axis in enumerate(symmetry_axes):
    # we set the radial trim such that we get 345 radial elements
    # we set the max ring difference such that we get 904 planes (similar to Siemens 837
    # planes in Span 11, but without spanning)
    lor_descriptor = scanners.SiemensMMRLORDescriptor(
        xp, dev, symmetry_axis=symmetry_axis, radial_trim = 52, max_ring_difference=7)
    views = xp.arange(0, lor_descriptor.num_views, num_subsets, device=dev)

    # setup a box like test image
    img_shape = [num_trans, num_trans, num_trans]
    img_shape[symmetry_axis] = num_ax
    img_shape = tuple(img_shape)
    n0, n1, n2 = img_shape

    # setup an image containing a square
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    sl = [
        slice(n0 // 4, 3 * n0 // 4, None),
        slice(n1 // 4, 3 * n1 // 4, None),
        slice(n2 // 4, 3 * n2 // 4, None)
    ]

    sl[symmetry_axis] = slice(0, img.shape[symmetry_axis], None)
    sl = tuple(sl)
    img[sl] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (-(xp.asarray(img.shape, dtype=xp.float32, device=dev) / 2) +
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
            img_fwd = parallelproj.joseph3d_fwd(
                xstart,
                xend,
                img,
                img_origin,
                voxel_size,
                threadsperblock=threadsperblock)
            t1 = time.time()

            # perform a complete backprojection
            ones = xp.ones(img_fwd.shape, dtype=xp.float32, device=dev)
            t2 = time.time()
            back_img = parallelproj.joseph3d_back(
                xstart,
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
                        'sinogram order': sinogram_order,
                        'symmetry axis': str(symmetry_axis),
                        'run': ir,
                        't forward (s)': t1 - t0,
                        't back (s)': t3 - t2
                    },
                    index=[0])
                df = pd.concat((df, tmp))

#----------------------------------------------------------------------------
print(df)
