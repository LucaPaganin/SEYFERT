import h5py
from pathlib import Path
from packaging.version import Version
from typing import Union, Dict
from seyfert.utils import general_utils as gu
from seyfert.cosmology.power_spectrum import PowerSpectrum
from seyfert.cosmology.parameter import PhysicalParameter
from seyfert.cosmology.cosmology import Cosmology
from seyfert.cosmology.c_ells import AngularCoefficientsCollector, AngularCoefficient
from seyfert.cosmology.redshift_density import RedshiftDensity
from seyfert.cosmology import weight_functions as wfs
from seyfert.cosmology import kernel_functions as kfs


def write_old_cosmo_file(cosmo, file):
    hf = h5py.File(file, mode='w')
    pmm = cosmo.power_spectrum

    hf.create_dataset(name='/cosmology/z_array', data=cosmo.z_grid)
    hf.create_dataset(name='/power_spectrum/k_array', data=pmm.k_grid)
    hf.create_dataset(name='/power_spectrum/p_mm_values/lin_p_mm_z_k', data=pmm.lin_p_mm_z_k)
    hf.create_dataset(name='/power_spectrum/p_mm_values/nonlin_p_mm_z_k', data=pmm.nonlin_p_mm_z_k)

    hf.close()


def read_v104_cosmo_file(file) -> "Cosmology":
    hf = h5py.File(file, mode='r')
    Ez = hf['/cosmology/dimensionless_hubble_parameter'][()]
    rz = hf['/cosmology/dimensionless_comoving_distance'][()]
    z = hf['/cosmology/z_array'][()]
    k = hf['/power_spectrum/k_array'][()]
    plin_zk = hf['/power_spectrum/p_mm_values/lin_p_mm_z_k'][()]
    pnonlin_zk = hf['/power_spectrum/p_mm_values/nonlin_p_mm_z_k'][()]

    growth_factor = None
    transfer_k = None
    if 'growth_factor_z' in hf:
        growth_factor = hf['growth_factor_z'][()]
    if 'transfer_function' in hf:
        transfer_k = hf['transfer_function'][()]

    hf.close()

    cosmo = Cosmology()
    cosmo.z_grid = z
    pmm = PowerSpectrum()
    pmm.z_grid = z
    pmm.k_grid = k
    pmm.lin_p_mm_z_k = plin_zk
    pmm.nonlin_p_mm_z_k = pnonlin_zk
    pmm.transfer_function = transfer_k
    cosmo.power_spectrum = pmm
    cosmo.dimensionless_hubble_array = Ez
    cosmo.dimensionless_comoving_distance_array = rz

    return cosmo


def read_v110_cosmo_file(file) -> "Cosmology":
    hf = h5py.File(file, mode='r')
    cosmo = Cosmology()
    cosmo.z_grid = hf['cosmology/z_grid'][()]
    cosmo.dimensionless_hubble_array = hf['cosmology/dimensionless_hubble_array'][()]
    cosmo.dimensionless_comoving_distance_array = hf['cosmology/dimensionless_comoving_distance_array'][()]

    pars = {
        name: dict(hf[f'cosmology/cosmological_parameters/{name}'].attrs)
        for name in hf[f'cosmology/cosmological_parameters']
    }
    cosmo.params = {}
    for name in pars:
        p = pars[name]
        cosmo.params[name] = PhysicalParameter(name=name, kind=PhysicalParameter.COSMO_PAR_STRING,
                                               fiducial=p['fiducial'],
                                               is_free_parameter=p['is_present'],
                                               current_value=p['current'])
    hf.close()
    return cosmo


def read_v104_cls_file(file):
    hf = h5py.File(file, mode='r')
    coll = AngularCoefficientsCollector()
    cls_grp = hf['/cls']
    dns_grp = hf['/density_functions']
    wfs_grp = hf['/weight_functions']
    kfs_grp = hf['/kernels']
    cl_keys = cls_grp.keys()
    for key in cl_keys:
        p1, p2 = gu.get_probes_from_comb_key(key)
        cl = AngularCoefficient(probe1=p1, probe2=p2)
        cl.c_lij = cls_grp[key]['c_lij'][()]
        cl.l_bin_centers = cls_grp[key]['l_bin_centers'][()]
        cl.l_bin_widths = cls_grp[key]['l_bin_widths'][()]
        cl.kernel = kfs.KernelFunction()
        cl.kernel.k_ijz = kfs_grp[key]['k_ijz'][()]
        w1 = wfs.weight_function_for_probe(probe=p1)
        w2 = wfs.weight_function_for_probe(probe=p2)
        w1.w_bin_z = wfs_grp[p1][()]
        w2.w_bin_z = wfs_grp[p2][()]
        w1.density = RedshiftDensity()
        w1.density.norm_density_iz = dns_grp[p1]['norm_density_i_z'][()]
        w2.density = RedshiftDensity()
        w2.density.norm_density_iz = dns_grp[p2]['norm_density_i_z'][()]
        cl.kernel.weight1 = w1
        cl.kernel.weight2 = w2
        coll.cl_dict[key] = cl
    return coll


def read_cls_file(file: "Union[str, Path]", version: "str"):
    v = Version(version)
    if v < Version('1.1.0'):
        coll = read_v104_cls_file(file)
    else:
        coll = AngularCoefficientsCollector.fromHDF5(file)
    return coll


def read_cosmo_file(file: "Union[str, Path]", version: "str", load_power_spectrum: "bool"):
    v = Version(version)
    if v < Version('1.1.0'):
        cosmo = read_v104_cosmo_file(file)
    elif v == Version('1.1.0'):
        cosmo = read_v110_cosmo_file(file)
    elif v > Version('1.1.0'):
        cosmo = Cosmology.fromHDF5(file, load_power_spectrum=load_power_spectrum)
    else:
        raise ValueError(f'Unrecognized version {v}')
    return cosmo


def read_dcl_file(file: "Union[str, Path]") -> "Dict":
    hf = h5py.File(file, mode='r')
    dcl_data = {}
    for key, entry in hf.items():
        if isinstance(entry, h5py.Dataset):
            dcl_data[key] = entry[()]
        elif isinstance(entry, h5py.Group):
            dcl_data[key] = entry["dc_lij"][()]
        else:
            raise TypeError(f"Invalid HDF5 entry type {type(entry)}")

    return dcl_data
