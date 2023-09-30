from __future__ import annotations

import enum
import abc
import array_api_compat.numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from types import ModuleType
from array_api_compat import get_namespace, device, to_device

class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""

    RVP = enum.auto()
    """[radial,view,plane]"""
    RPV = enum.auto()
    """[radial,plane,view]"""
    VRP = enum.auto()
    """[view,radial,plane]"""
    VPR = enum.auto()
    """[view,plane,radial]"""
    PRV = enum.auto()
    """[plane,radial,view]"""
    PVR = enum.auto()
    """[plane,view,radial]"""

class PETScannerModule(abc.ABC):

    def __init__(
            self,
            xp: ModuleType,
            dev : str,
            num_lor_endpoints: int,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """abstract base class for PET scanner module

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._xp = xp
        self._dev = dev
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = self.xp.arange(num_lor_endpoints, device = self.dev)

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = self.xp.eye(4, device = self.dev)
        else:
            self._affine_transformation_matrix = affine_transformation_matrix

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self._xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self._dev

    @property
    def num_lor_endpoints(self) -> int:
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> npt.NDArray:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        npt.NDArray
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> npt.NDArray:
        """4x4 affine transformation matrix

        Returns
        -------
        npt.NDArray
        """
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints
        """
        if inds is None:
            inds = self.lor_endpoint_numbers
        raise NotImplementedError

    def get_lor_endpoints(self,
                          inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints including an affine transformation
        """

        raw_lor_endpoints = self.get_raw_lor_endpoints(inds)

        tmp = self.xp.ones((raw_lor_endpoints.shape[0], 4), device = self.dev)
        tmp[:,:-1] = raw_lor_endpoints

        return (tmp @ self.affine_transformation_matrix.T)[:, :3]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           annotation_fontsize: float = 0,
                           annotation_prefix: str = '',
                           annotation_offset: int = 0,
                           transformed: bool = True,
                           **kwargs) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : plt.Axes
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        # convert to numpy array
        all_lor_endpoints = np.asarray(to_device(x, 'cpu'))

        ax.scatter(all_lor_endpoints[:, 0], all_lor_endpoints[:, 1],
                   all_lor_endpoints[:, 2], **kwargs)

        ax.set_box_aspect([
            ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')
        ])

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(all_lor_endpoints[i, 0],
                        all_lor_endpoints[i, 1],
                        all_lor_endpoints[i, 2],
                        f'{annotation_prefix}{i+annotation_offset}',
                        fontsize=annotation_fontsize)

class RegularPolygonPETScannerModule(PETScannerModule):

    def __init__(
            self,
            xp: ModuleType,
            dev : str,
            radius: float,
            num_sides: int,
            num_lor_endpoints_per_side: int,
            lor_spacing: float,
            ax0: int = 2,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """regular Polygon PET scanner module

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        device: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_sides: int
            number of LOR endpoints per side
        lor_spacing : float
            spacing between the LOR endpoints in the polygon direction
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._ax0 = ax0
        self._ax1 = ax1
        self._lor_spacing = lor_spacing
        super().__init__(xp, dev, num_sides * num_lor_endpoints_per_side,
                         affine_transformation_matrix)

    @property
    def radius(self) -> float:
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float:
        """spacing between the LOR endpoints in a module along the polygon

        Returns
        -------
        float
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        if inds is None:
            inds = self.lor_endpoint_numbers

        side = inds // self.num_lor_endpoints_per_side
        tmp = inds - side * self.num_lor_endpoints_per_side
        tmp = self.xp.astype(tmp,float) - (self.num_lor_endpoints_per_side / 2 - 0.5)

        phi = 2 * np.pi * self.xp.astype(side, float) / self.num_sides

        lor_endpoints = self.xp.zeros((self.num_lor_endpoints, 3), device = self.dev)
        lor_endpoints[:, self.ax0] = np.cos(phi) * self.radius - np.sin(
            phi) * self.lor_spacing * tmp
        lor_endpoints[:, self.ax1] = np.sin(phi) * self.radius + np.cos(
            phi) * self.lor_spacing * tmp

        return lor_endpoints


if __name__ == '__main__':
    import numpy.array_api as xp
    #import array_api_compat.torch as xp
    dev = 'cpu'

    mod =  RegularPolygonPETScannerModule(
            xp,
            dev,
            radius = 350.,
            num_sides = 28,
            num_lor_endpoints_per_side = 16,
            lor_spacing = 4.)

    x = mod.get_lor_endpoints()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mod.show_lor_endpoints(ax)
    fig.show()



#class ModularizedPETScannerGeometry:
#    """description of a PET scanner geometry consisting of LOR endpoint modules"""
#
#    def __init__(self,
#                 modules: tuple[mods.PETScannerModule],
#                 xp: ModuleType | None = None) -> None:
#        """
#        Parameters
#        ----------
#        modules : tuple[PETScannerModule]
#            a tuple of scanner modules
#        xp : types.ModuleType | None, optional default None
#            module indicating whether to store all LOR endpoints as numpy as cupy array
#            default None means that numpy is used
#        """
#
#        # member variable that determines whether we want to use
#        # a numpy or cupy array to store the array of all lor endpoints
#        if xp is None:
#            self._xp = np
#        else:
#            self._xp = xp
#
#        self._modules = modules
#        self._num_modules = len(self._modules)
#        self._num_lor_endpoints_per_module = self._xp.array(
#            [x.num_lor_endpoints for x in self._modules])
#        self._num_lor_endpoints = self._num_lor_endpoints_per_module.sum()
#        if self._xp.__name__ not in ['numpy', 'cupy']:
#            raise ValueError('xp must be numpy or cupy module')
#
#        self.setup_all_lor_endpoints()
#
#    def setup_all_lor_endpoints(self) -> None:
#        """calculate the position of all lor endpoints by iterating over
#           the modules and calculating the transformed coordinates of all
#           module endpoints
#        """
#        self._all_lor_endpoints_index_offset = self._xp.cumsum(
#            self._xp.pad(self._num_lor_endpoints_per_module,
#                         (1, 0)))[:self._num_modules]
#
#        self._all_lor_endpoints = self._xp.vstack(
#            [x.get_lor_endpoints() for x in self._modules])
#
#        self._all_lor_endpoints_module_number = self._xp.repeat(
#            self._xp.arange(self._num_modules),
#            self._num_lor_endpoints_per_module.tolist())
#
#    @property
#    def modules(self) -> tuple[mods.PETScannerModule]:
#        """tuple of modules defining the scanner"""
#        return self._modules
#
#    @property
#    def num_modules(self) -> int:
#        """the number of modules defining the scanner"""
#        return self._num_modules
#
#    @property
#    def num_lor_endpoints_per_module(self) -> npt.NDArray | cpt.NDArray:
#        """numpy array showing how many LOR endpoints are in every module"""
#        return self._num_lor_endpoints_per_module
#
#    @property
#    def num_lor_endpoints(self) -> int:
#        """the total number of LOR endpoints in the scanner"""
#        return self._num_lor_endpoints
#
#    @property
#    def all_lor_endpoints_index_offset(self) -> npt.NDArray | cpt.NDArray:
#        """the offset in the linear (flattend) index for all LOR endpoints"""
#        return self._all_lor_endpoints_index_offset
#
#    @property
#    def all_lor_endpoints_module_number(self) -> npt.NDArray | cpt.NDArray:
#        """the module number of all LOR endpoints"""
#        return self._all_lor_endpoints_module_number
#
#    @property
#    def all_lor_endpoints(self) -> npt.NDArray | cpt.NDArray:
#        """the world coordinates of all LOR endpoints"""
#        return self._all_lor_endpoints
#
#    @property
#    def xp(self) -> types.ModuleType:
#        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
#        return self._xp
#
#    @xp.setter
#    def xp(self, value: types.ModuleType):
#        """set the module to use for storing all LOR endpoints"""
#        self._xp = value
#        self.setup_all_lor_endpoints()
#
#    def linear_lor_endpoint_index(
#        self, module: npt.NDArray | cpt.NDArray,
#        index_in_module: npt.NDArray | cpt.NDArray
#    ) -> npt.NDArray | cpt.NDArray:
#        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index
#
#        Parameters
#        ----------
#        module : npt.NDArray
#            containing module numbers
#        index_in_module : npt.NDArray
#            containing index in modules
#
#        Returns
#        -------
#        npt.NDArray
#            the flattened LOR endpoint index
#        """
#        if (self._xp.__name__ == 'cupy') and isinstance(
#                index_in_module, np.ndarray):
#            index_in_module = self._xp.asarray(index_in_module)
#
#        return self.all_lor_endpoints_index_offset[module] + index_in_module
#
#    def get_lor_endpoints(
#            self, module: npt.NDArray,
#            index_in_module: npt.NDArray) -> npt.NDArray | cpt.NDArray:
#        """get the coordinates for LOR endpoints defined by module and index in module
#
#        Parameters
#        ----------
#        module : npt.NDArray
#            the module number of the LOR endpoints
#        index_in_module : npt.NDArray
#            the index in module number of the LOR endpoints
#
#        Returns
#        -------
#        npt.NDArray | cpt.NDArray
#            the 3 world coordinates of the LOR endpoints
#        """
#        return self.all_lor_endpoints[
#            self.linear_lor_endpoint_index(module, index_in_module), :]
#
#    def show_lor_endpoints(self,
#                           ax: plt.Axes,
#                           show_linear_index: bool = True,
#                           **kwargs) -> None:
#        """show all LOR endpoints in a 3D plot
#
#        Parameters
#        ----------
#        ax : plt.Axes
#            a 3D matplotlib axes
#        show_linear_index : bool, optional
#            annotate the LOR endpoints with the linear LOR endpoint index
#        **kwargs : keyword arguments
#            passed to show_lor_endpoints() of the scanner module
#        """
#        for i, module in enumerate(self.modules):
#            if show_linear_index:
#                offset = self.all_lor_endpoints_index_offset[i]
#                prefix = f''
#            else:
#                offset = 0
#                prefix = f'{i},'
#
#            module.show_lor_endpoints(ax,
#                                      annotation_offset=offset,
#                                      annotation_prefix=prefix,
#                                      **kwargs)
#
#
#class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):
#    """description of a PET scanner geometry consisting stacked regular polygons"""
#
#    def __init__(self,
#                 radius: float,
#                 num_sides: int,
#                 num_lor_endpoints_per_side: int,
#                 lor_spacing: float,
#                 num_rings: int,
#                 ring_positions: npt.NDArray,
#                 symmetry_axis: int,
#                 xp: types.ModuleType | None = None) -> None:
#        """
#        Parameters
#        ----------
#        radius : float
#            radius of the scanner
#        num_sides : int
#            number of sides (faces) of each regular polygon
#        num_lor_endpoints_per_side : int
#            number of LOR endpoints in each side (face) of each polygon
#        lor_spacing : float
#            spacing between the LOR endpoints in each side
#        num_rings : int
#            the number of rings (regular polygons)
#        ring_positions : npt.NDArray
#            1D array with the coordinate of the rings along the ring axis
#        symmetry_axis : int
#            the ring axis (0,1,2)
#        xp : types.ModuleType | None, optional default None
#            numpy or cupy module used to store the coordinates of all LOR endpoints, by default np
#            None means that numpy is used
#        """
#
#        self._radius = radius
#        self._num_sides = num_sides
#        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
#        self._num_rings = num_rings
#        self._lor_spacing = lor_spacing
#        self._symmetry_axis = symmetry_axis
#
#        if symmetry_axis == 0:
#            self._ax0 = 2
#            self._ax1 = 1
#        elif symmetry_axis == 1:
#            self._ax0 = 0
#            self._ax1 = 2
#        elif symmetry_axis == 2:
#            self._ax0 = 1
#            self._ax1 = 0
#
#        modules = []
#
#        for ring in range(num_rings):
#            aff_mat = np.eye(4)
#            aff_mat[symmetry_axis, -1] = ring_positions[ring]
#
#            modules.append(
#                mods.RegularPolygonPETScannerModule(
#                    radius,
#                    num_sides,
#                    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
#                    lor_spacing=lor_spacing,
#                    affine_transformation_matrix=aff_mat,
#                    ax0=self._ax0,
#                    ax1=self._ax1))
#
#        modules = tuple(modules)
#        super().__init__(modules, xp)
#
#        self._all_lor_endpoints_index_in_ring = self._xp.arange(
#            self.num_lor_endpoints
#        ) - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[
#            0]
#
#    @property
#    def radius(self) -> float:
#        """radius of the scanner"""
#        return self._radius
#
#    @property
#    def num_sides(self) -> int:
#        """number of sides (faces) of each polygon"""
#        return self._num_sides
#
#    @property
#    def num_lor_endpoints_per_side(self) -> int:
#        """number of LOR endpoints per side (face) in each polygon"""
#        return self._num_lor_endpoints_per_side
#
#    @property
#    def num_rings(self) -> int:
#        """number of rings (regular polygons)"""
#        return self._num_rings
#
#    @property
#    def lor_spacing(self) -> float:
#        """the spacing between the LOR endpoints in every side (face) of each polygon"""
#        return self._lor_spacing
#
#    @property
#    def symmetry_axis(self) -> int:
#        """The symmetry axis. Also called axial (or ring) direction."""
#        return self._symmetry_axis
#
#    @property
#    def all_lor_endpoints_ring_number(self) -> npt.NDArray:
#        """the ring (regular polygon) number of all LOR endpoints"""
#        return self._all_lor_endpoints_module_number
#
#    @property
#    def all_lor_endpoints_index_in_ring(self) -> npt.NDArray:
#        """the index withing the ring (regular polygon) number of all LOR endpoints"""
#        return self._all_lor_endpoints_index_in_ring
#
#    @property
#    def num_lor_endpoints_per_ring(self) -> int:
#        """the number of LOR endpoints per ring (regular polygon)"""
#        return int(self._num_lor_endpoints_per_module[0])
#
#
#class GEDiscoveryMI(RegularPolygonPETScannerGeometry):
#
#    def __init__(self,
#                 num_rings: int = 36,
#                 symmetry_axis: int = 2,
#                 xp: types.ModuleType = np):
#
#        ring_positions = 5.31556 * np.arange(num_rings) + (
#            np.arange(num_rings) // 9) * 2.8
#        ring_positions -= 0.5 * ring_positions.max()
#        super().__init__(radius=0.5 * (744.1 + 2 * 8.51),
#                         num_sides=34,
#                         num_lor_endpoints_per_side=16,
#                         lor_spacing=4.03125,
#                         num_rings=num_rings,
#                         ring_positions=ring_positions,
#                         symmetry_axis=symmetry_axis,
#                         xp=xp)




#class PETCoincidenceDescriptor(abc.ABC):
#    """abstract base class to describe which modules / indices in modules of a 
#       modularized PET scanner are in coincidence; defining geometrical LORs"""
#
#    def __init__(self,
#                 scanner: scanners.ModularizedPETScannerGeometry) -> None:
#        """
#        Parameters
#        ----------
#        scanner : ModularizedPETScannerGeometry
#            a modularized PET scanner 
#        """
#        self._scanner = scanner
#
#    @property
#    def scanner(self) -> scanners.ModularizedPETScannerGeometry:
#        """the scanner for which coincidences are described"""
#        return self._scanner
#
#    #-------------------------------------------------------------------
#    #-------------------------------------------------------------------
#    # abstract methods
#
#    @property
#    @abc.abstractmethod
#    def num_lors(self) -> int:
#        """the total number of geometrical LORs 
#        """
#        raise NotImplementedError
#
#    @abc.abstractmethod
#    def get_modules_and_indices_in_coincidence(
#            self, module: int, index_in_module: int) -> npt.NDArray:
#        """ return (N,2) array of two integers showing which module/index_in_module combinations
#            are in coincidence with the given input module / index_in_module
#
#        Parameters
#        ----------
#        module : int
#            the module number
#        index_in_module : int
#            the (LOR endpoint) index in the module
#
#        Returns
#        -------
#        npt.NDArray
#            (N,2) array of two integers showing which module/index_in_module
#        """
#
#    @abc.abstractmethod
#    def get_lor_indices(
#        self,
#        linear_lor_indices: None | slice | npt.NDArray = None
#    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
#        """ mapping that maps the linear LOR index to the 4 1D arrays
#            representing start_module, start_index_in_module, end_module,
#            end_index_in_module
#
#        Parameters
#        ----------
#        linear_lor_indices : None | slice | npt.NDArray, optional
#            containing the linear (flattened) indices of geometrical LORs, by default None
#
#        Returns
#        -------
#        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
#            start_module, start_index_in_module, end_module, end_index_in_module
#        """
#
#    #-------------------------------------------------------------------
#    #-------------------------------------------------------------------
#
#    def setup_lor_lookup_table(self) -> None:
#        """setup a lookup table for the start and end modules / indecies in module for all 
#           geometrical LORs
#        """
#        for mod, num_lor_endpoints in enumerate(
#                self.scanner.num_lor_endpoints_per_module):
#            for lor in range(num_lor_endpoints):
#                tmp = self.get_modules_and_indices_in_coincidence(mod, lor)
#                # make sure we do not LORs twice
#                tmp = tmp[tmp[:, 0] >= mod]
#
#                if mod == 0 and lor == 0:
#                    self._lor_start_module_index = np.repeat(np.array(
#                        [[mod, lor]], dtype=np.uint16),
#                                                             tmp.shape[0],
#                                                             axis=0)
#                    self._lor_end_module_index = tmp.copy().astype(np.uint16)
#                else:
#                    self._lor_start_module_index = np.vstack(
#                        (self._lor_start_module_index,
#                         np.repeat(np.array([[mod, lor]]),
#                                   tmp.shape[0],
#                                   axis=0)))
#                    self._lor_end_module_index = np.vstack(
#                        (self._lor_end_module_index, tmp))
#
#    def show_all_lors_for_endpoint(self,
#                                   ax: plt.Axes,
#                                   module: int,
#                                   index_in_module: int,
#                                   lw: float = 0.2,
#                                   **kwargs) -> None:
#        """show all geometrical LORs for a given LOR endpoint
#
#        Parameters
#        ----------
#        ax : plt.Axes
#            a 3D matplotlib axes
#        module : int
#            the module number
#        index_in_module : int
#            the index in the module
#        lw : float, optional
#            line width, by default 0.2
#        **kwargs : 
#            keyword arguments passed to Line3DCollection
#        """
#
#        tmp = self.get_modules_and_indices_in_coincidence(
#            module, index_in_module)
#
#        end_mod = tmp[:, 0]
#        end_ind = tmp[:, 1]
#
#        start_mod = np.full(end_mod.shape[0], module)
#        start_ind = np.full(end_mod.shape[0], index_in_module)
#
#        p1s = self.scanner.get_lor_endpoints(start_mod, start_ind)
#        p2s = self.scanner.get_lor_endpoints(end_mod, end_ind)
#
#        # get_lor_endpoints can return a numpy or cupy array
#        # it scanner uses cupy arrays, we have to convert them into numpy arrays
#        if self.scanner.xp.__name__ == 'cupy':
#            p1s = self.scanner.xp.asnumpy(p1s)
#            p2s = self.scanner.xp.asnumpy(p2s)
#
#        ls = np.hstack([p1s, p2s]).copy()
#        ls = ls.reshape((-1, 2, 3))
#        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
#        ax.add_collection(lc)
#
#    def show_lors(self,
#                  ax: plt.Axes,
#                  lors: None | npt.NDArray,
#                  lw: float = 0.2,
#                  **kwargs) -> None:
#        """show a given set of LORs
#
#        Parameters
#        ----------
#        ax : plt.Axes
#            a 3D matplotlib axes
#        lors : None | npt.NDArray
#            the linear (flattened) index of the geometrical LORs to show
#            None means all geometrical LORs are shown
#        lw : float, optional
#            linewidth, by default 0.2
#        **kwargs : 
#            keyword arguments passed to Line3DCollection
#        """
#        start_mod, start_ind, end_mod, end_ind = self.get_lor_indices(lors)
#        p1s = self.scanner.get_lor_endpoints(start_mod, start_ind)
#        p2s = self.scanner.get_lor_endpoints(end_mod, end_ind)
#
#        # get_lor_endpoints can return a numpy or cupy array
#        # it scanner uses cupy arrays, we have to convert them into numpy arrays
#        if self.scanner.xp.__name__ == 'cupy':
#            p1s = self.scanner.xp.asnumpy(p1s)
#            p2s = self.scanner.xp.asnumpy(p2s)
#
#        ls = np.hstack([p1s, p2s]).copy()
#        ls = ls.reshape((-1, 2, 3))
#        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
#        ax.add_collection(lc)
#
#