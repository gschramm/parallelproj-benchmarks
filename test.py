import matplotlib.pyplot as plt
import scanners

if __name__ == '__main__':
    import numpy.array_api as xp
    #import array_api_compat.torch as xp
    #import array_api_compat.cupy as xp

    dev = 'cpu'
    #dev = 'cuda'

    coinc_des = scanners.GEDiscoveryMILORDescriptor(xp, dev, symmetry_axis=2)

    views = xp.arange(0, coinc_des.num_views, 34, device=dev)
    xs, xe = coinc_des.get_lor_coordinates(
        views,
        sinogram_order=scanners.SinogramSpatialAxisOrder['RPV'])

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #coinc_des.scanner.show_lor_endpoints(ax)
    #coinc_des.show_views(ax, xp.asarray([0], device=dev),
    #                     xp.asarray([8], device=dev))
    #fig.show()
