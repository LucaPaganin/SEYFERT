import os
from typing import Dict
import json
import datetime

from seyfert import VERSION


class JSONForecastConfig:
    def __init__(self, file=None, data=None):
        if data is None:
            data = {}
        self.data = data

        if not self.data and file is not None:
            self.loadJSON(file)

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return repr(self.data)

    def loadJSON(self, file):
        with open(file) as jsf:
            self.data = json.load(jsf)

    def toJSON(self, file):
        with open(file, mode='w') as jsf:
            self.metadata.update({
                "author": os.getenv("USER"),
                "creation_date": datetime.datetime.now().isoformat(),
                "version": VERSION
            })
            json.dump(self.data, jsf, indent=4)

    @property
    def metadata(self) -> "Dict":
        return self.data["metadata"]

    @property
    def scenario(self) -> "str":
        return self.synthetic_opts["scenario"]

    @property
    def n_sp_bins(self) -> "int":
        return self.synthetic_opts["n_sp_bins"]

    @property
    def synthetic_opts(self) -> "Dict":
        return self.data["synthetic_opts"]

    @property
    def survey(self):
        return self.data["survey"]

    @property
    def f_sky(self) -> "float":
        return self.survey["f_sky"]

    @property
    def shot_noise_file(self):
        return self.survey["shot_noise_file"]

    @shot_noise_file.setter
    def shot_noise_file(self, value):
        self.survey["shot_noise_file"] = value

    @property
    def cosmology(self):
        return self.data["cosmology"]

    @property
    def probes(self) -> "Dict":
        return self.data["probes"]

    @property
    def derivative_settings(self):
        return self.data["derivative_settings"]

    @property
    def WL(self):
        return self.probes["WL"]

    @property
    def GCph(self):
        return self.probes["GCph"]

    @property
    def GCsp(self):
        return self.probes["GCsp"]

    @property
    def Void(self):
        return self.probes["V"]

    def getConfigID(self) -> "str":
        conf_id_parts = []
        for key, value in self.synthetic_opts.items():
            if isinstance(value, bool):
                if value:
                    conf_id_parts.append(key)
            else:
                conf_id_parts.append(f"{key}_{value}")

        if conf_id_parts:
            conf_id = "__".join(conf_id_parts)
        else:
            conf_id = None

        return conf_id

