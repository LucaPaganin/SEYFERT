import numpy as np
from typing import Union, Dict, List
import json
from pathlib import Path
import logging
from seyfert.base_structs.generic_dict import GenericDictInterface
from seyfert.utils.encoders import PhysParJSONEncoder

logger = logging.getLogger(__name__)


class ParameterError(Exception):
    pass


class PhysicalParameter:
    name: "str"
    fiducial: "float"
    current_value: "float"
    par_type: "str"
    is_free_parameter: "bool"
    probe: "str"
    description: "str"
    derivative_method: "str"
    COSMO_PAR_STRING = "CosmologicalParameter"
    NUISANCE_PAR_STRING = "NuisanceParameter"

    def __init__(self, name: "str" = None, fiducial: float = None, current_value: float = None, kind: str = None,
                 probe: str = None, is_free_parameter: bool = None, stem_factor: float = 1.0, description: str = "",
                 units: "str" = "None", derivative_method: "str" = "SteM"):
        self.name = name
        self.fiducial = fiducial
        self.current_value = current_value
        self.probe = probe
        self.kind = kind
        self.stem_factor = stem_factor
        self.is_free_parameter = is_free_parameter
        self.description = description
        self.units = units
        self.derivative_method = derivative_method

        if not isinstance(self.name, str):
            raise TypeError(f'Name must be str, not {type(self.name)}')
        if not isinstance(self.fiducial, float):
            raise TypeError(f'Fiducial must be float, not {type(self.fiducial)}')
        if not isinstance(self.is_free_parameter, (bool, np.bool_)):
            raise TypeError(f'presence flag must be bool, not {type(self.is_free_parameter)}')
        if self.kind not in {self.COSMO_PAR_STRING, self.NUISANCE_PAR_STRING}:
            raise ValueError(f'kind must be {self.COSMO_PAR_STRING} or {self.NUISANCE_PAR_STRING}, '
                             f'not {self.kind}')
        if self.is_nuisance and self.probe is None:
            raise ValueError(f'probe must be specified for nuisance parameters')
        if self.current_value is None:
            self.current_value = self.fiducial

    @classmethod
    def fromJSON(cls, json_file: Union[str, Path]) -> "PhysicalParameter":
        with open(json_file, 'r') as jsf:
            data = json.load(jsf)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: "Dict") -> "PhysicalParameter":
        return cls(**data)

    @classmethod
    def createNuisanceParameter(cls, **kwargs) -> "PhysicalParameter":
        return cls(kind=cls.NUISANCE_PAR_STRING, **kwargs)

    @classmethod
    def createCosmologicalParameter(cls, **kwargs) -> "PhysicalParameter":
        return cls(kind=cls.COSMO_PAR_STRING, **kwargs)

    def to_dict(self) -> "Dict[str, Union[str, float, bool]]":
        data = {
            'name': self.name,
            'fiducial': self.fiducial,
            'current_value': self.current_value,
            'kind': self.kind,
            'is_free_parameter': self.is_free_parameter,
            'stem_factor': self.stem_factor,
            'derivative_method': self.derivative_method,
            'units': self.units
        }
        if self.is_nuisance:
            data['probe'] = self.probe

        return data

    def to_JSON(self) -> "str":
        return json.dumps(self.to_dict(), cls=PhysParJSONEncoder, indent=2)

    @property
    def is_cosmological(self) -> "bool":
        return self.kind == self.COSMO_PAR_STRING

    @property
    def is_nuisance(self) -> "bool":
        return self.kind == self.NUISANCE_PAR_STRING

    @property
    def is_galaxy_bias_parameter(self) -> "bool":
        return self.is_nuisance and self.name.startswith("bg")

    def resetCurrentValueToFiducial(self):
        self.current_value = self.fiducial

    def computeValueForDisplacement(self, eps: "float") -> "float":
        delta_p = eps * self.stem_factor
        if self.fiducial != 0:
            delta_p *= self.fiducial
        return self.fiducial + delta_p

    def computeSteMValues(self, stem_eps_arr: "np.ndarray") -> "np.ndarray":
        values = [self.computeValueForDisplacement(eps) for eps in stem_eps_arr]
        values.insert(len(values) // 2, self.fiducial)
        return np.array(values)

    def updateValueForDisplacement(self, eps: "float"):
        self.current_value = self.computeValueForDisplacement(eps)

    def __repr__(self) -> str:
        s = f'{self.kind}, name: {self.name}, is_free: {self.is_free_parameter}, ' \
            f'fiducial: {self.fiducial}, current: {self.current_value}, diff_method: {self.derivative_method}, ' \
            f'units: {self.units}'
        if self.probe is not None:
            s += f', probe: {self.probe}'
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: "PhysicalParameter") -> "bool":
        return all([
            self.name == other.name, self.fiducial == other.fiducial, self.kind == other.kind,
            self.current_value == other.current_value
        ])


class PhysicalParametersCollection(GenericDictInterface[str, PhysicalParameter]):
    stem_steps: "np.ndarray"
    stem_disps: "np.ndarray"
    stem_dict: "Dict[int, float]"

    def __init__(self, base_stem_disps: "np.ndarray" = None, is_universe_flat: "bool" = None, **kwargs):
        super(PhysicalParametersCollection, self).__init__(**kwargs)
        self.stem_steps = None
        self.stem_disps = None
        self.stem_dict = None
        self.inv_stem_dict = None
        self.is_universe_flat = is_universe_flat

        if base_stem_disps is not None:
            self.loadStemDisplacements(base_stem_disps)

    @property
    def params(self) -> "Dict[str, PhysicalParameter]":
        return self._base_dict

    @property
    def cosmological_parameters(self) -> "Dict[str, PhysicalParameter]":
        return {name: par for name, par in self.items() if par.is_cosmological}

    @property
    def nuisance_parameters(self) -> "Dict[str, PhysicalParameter]":
        return {name: par for name, par in self.items() if par.is_nuisance}

    @property
    def free_physical_parameters(self) -> "Dict[str, PhysicalParameter]":
        return {name: par for name, par in self.items() if par.is_free_parameter}

    @property
    def free_cosmological_parameters(self) -> "Dict[str, PhysicalParameter]":
        return {k: par for k, par in self.cosmological_parameters.items() if par.is_free_parameter}

    @property
    def cosmo_pars_fiducials(self) -> "Dict[str, float]":
        return {k: par.fiducial for k, par in self.cosmological_parameters.items()}

    @property
    def free_cosmo_pars_fiducials(self) -> Dict[str, float]:
        return {key: par.fiducial for key, par in self.free_cosmological_parameters.items()}

    @property
    def cosmo_pars_current_values(self) -> Dict[str, float]:
        return {k: par.current_value for k, par in self.cosmological_parameters.items()}

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict], only_cosmological: "bool" = False) -> "PhysicalParametersCollection":
        data_dict = cls.getParamsDictFromDictList(dict_list, only_cosmological=only_cosmological)
        base_stem_disps = dict_list[-1]['base_stem_disps']
        is_universe_flat = dict_list[-1]['is_universe_flat']
        return cls(data_dict=data_dict, base_stem_disps=base_stem_disps, is_universe_flat=is_universe_flat)

    @classmethod
    def fromJSON(cls, file: "Union[str, Path]", only_cosmological: "bool" = False) -> "PhysicalParametersCollection":
        with open(file, 'r') as jsf:
            dict_list = json.load(jsf)
        return cls.from_dict_list(dict_list, only_cosmological=only_cosmological)

    def readJSON(self, file: "Union[str, Path]", only_cosmological: "bool" = False) -> "None":
        with open(file, 'r') as jsf:
            data = json.load(jsf)
        self.update(self.getParamsDictFromDictList(data, only_cosmological=only_cosmological))
        self.loadStemDisplacements(data[-1]['base_stem_disps'])
        self.is_universe_flat = data[-1]['is_universe_flat']

    def writeJSON(self, file: "Union[str, Path]") -> "None":
        with open(file, 'w') as jsf:
            dict_list = [par.to_dict() for par in self.values()]
            dict_list.append({
                'base_stem_disps': list(self.stem_disps[self.stem_disps > 0]),
                'is_universe_flat': self.is_universe_flat
            })
            json.dump(dict_list, jsf, indent=2)

    def loadStemDisplacements(self, base_stem_disp: "Union[np.ndarray, List]") -> "None":
        if not isinstance(base_stem_disp, np.ndarray):
            base_stem_disp = np.array(base_stem_disp, dtype=float)
        self.stem_disps = np.concatenate([-base_stem_disp, base_stem_disp])
        self.stem_disps.sort()
        self.stem_steps = np.r_[-len(base_stem_disp):0, 1:len(base_stem_disp)+1]
        self.stem_dict = dict(zip(self.stem_steps, self.stem_disps))
        self.inv_stem_dict = {value: key for key, value in self.stem_dict.items()}

    def getNuisanceParametersForProbe(self, probe: "str") -> "Dict[str, PhysicalParameter]":
        return {name: par for name, par in self.items() if par.is_nuisance and par.probe == probe}

    def getFreeNuisanceParametersForProbe(self, probe: "str") -> "Dict[str, PhysicalParameter]":
        return {
            name: par for name, par in self.getNuisanceParametersForProbe(probe).items()
            if par.is_free_parameter
        }

    def getFreePhysicalParametersForProbe(self, probe: "str") -> "Dict[str, PhysicalParameter]":
        free_phys_pars = {}
        free_phys_pars.update(self.free_cosmological_parameters)
        free_phys_pars.update(self.getFreeNuisanceParametersForProbe(probe))

        return free_phys_pars

    @staticmethod
    def getParamsDictFromDictList(dict_list: "List[Dict]",
                                  only_cosmological: "bool" = False) -> "Dict[str, PhysicalParameter]":
        d = {}
        for param_dict in dict_list[:-1]:
            param = PhysicalParameter.from_dict(param_dict)
            if only_cosmological:
                if param.is_cosmological:
                    d[param.name] = param
                else:
                    logger.info(f'only cosmological flag active, skipping parameter {param}')
            else:
                d[param.name] = param
        return d

    def updatePhysicalParametersForDvarStep(self, dvar: "str" = None, step: "int" = None) -> "None":
        if dvar == 'central':
            logger.info('Fiducial cosmology resetting all parameters to fiducial values')
            self.resetPhysicalParametersToFiducial()
        else:
            if not self[dvar].is_free_parameter:
                raise ParameterError(f'Parameter {dvar} is not present but you are trying to vary it')

            self[dvar].updateValueForDisplacement(self.stem_dict[step])

            if self.is_universe_flat and dvar in {'Omm', 'OmDE'}:
                Omm = self["Omm"]
                OmDE = self["OmDE"]
                if Omm.fiducial + OmDE.fiducial != 1:
                    raise ParameterError(f'Fiducials Omm and OmDE must sum to 1 for flat Universe, not '
                                         f'{Omm.fiducial + OmDE.fiducial}')
                if Omm.is_free_parameter and OmDE.is_free_parameter:
                    raise ParameterError('Omm and OmDE cannot be both free in flat cosmology')
                logger.info(f'{dvar} has been varied in flat Universe')
                if dvar == "Omm" and not OmDE.is_free_parameter:
                    logger.info(f'Setting OmDE current value to 1-Omm = {1-Omm.current_value}')
                    OmDE.current_value = 1 - Omm.current_value
                elif dvar == "OmDE" and not Omm.is_free_parameter:
                    logger.info(f'Setting Omm current value to 1-OmDE = {1 - OmDE.current_value}')
                    Omm.current_value = 1 - OmDE.current_value

    def resetPhysicalParametersToFiducial(self) -> "None":
        for par in self:
            self[par].resetCurrentValueToFiducial()

    def computePhysParSTEMValues(self, dvar: "str" = None) -> "np.ndarray":
        return self[dvar].computeSteMValues(self.stem_disps)

    def __repr__(self) -> "str":
        return json.dumps(self._base_dict, indent=2, cls=PhysParJSONEncoder)
