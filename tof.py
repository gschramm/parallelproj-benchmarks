from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TOFParameters:
    """
    time of flight (TOF) parameters

    num_tofbins: int
        number of time of flight bins
    tofbin_width: float
        width of the TOF bin in spatial units
    sigma_tof: float
        standard deviation of Gaussian TOF kernel in spatial units
    num_sigmas: float
        number of sigmas after which TOF kernel is truncated
    tofcenter_offset: float
        offset of center of central TOF bin from LOR center in spatial units
    """
    num_tofbins: int
    tofbin_width: float
    sigma_tof: float
    num_sigmas: float = 3.
    tofcenter_offset: float = 0


ge_discovery_mi_tof_parameters = TOFParameters(
    num_tofbins=29,
    tofbin_width=13 * 0.01302 * 299.792 /
    2,  # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    sigma_tof=(299.792 / 2) *
    (0.385 / 2.355),  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas=3,
    tofcenter_offset=0)