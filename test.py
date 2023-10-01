import matplotlib.pyplot as plt
import scanners

if __name__ == '__main__':
    import numpy.array_api as xp
    #import array_api_compat.torch as xp
    #import array_api_compat.cupy as xp

    #dev = 'cpu'
    dev = 'cuda'

    coinc_des = scanners.GEDiscoveryMILORDescriptor(xp, dev)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coinc_des.scanner.show_lor_endpoints(ax)
    xs, xe = coinc_des.get_lor_coordinates(
        xp.asarray([0, 8], device=dev),
        sinogram_order=scanners.SinogramSpatialAxisOrder.RVP)
    coinc_des.show_views(ax, xp.asarray([0], device=dev),
                         xp.asarray([8], device=dev))
    fig.show()
