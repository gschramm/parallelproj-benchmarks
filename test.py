import matplotlib.pyplot as plt
import scanners

if __name__ == '__main__':
    #import numpy.array_api as xp
    import array_api_compat.torch as xp
    #import array_api_compat.cupy as xp

    dev = 'cpu'
    #dev = 'cuda'

    coinc_des = scanners.GEDiscoveryMICoincidenceDescriptor(xp, dev)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coinc_des.scanner.show_lor_endpoints(ax)
    coinc_des.show_views(ax, xp.asarray([0]), xp.asarray([8]))
    fig.show()
