import sys
from typing import List, Tuple, Dict
from pathlib import Path
import datetime
import re
from functools import partial
from PyQt5.QtWidgets import QStackedWidget, QWidget, QDialog
from PyQt5 import QtCore, QtWidgets

from .gui_core import Ui_SeyfertConfigGUI
from seyfert.file_io.json_io import JSONForecastConfig
from seyfert.utils.formatters import str_to_bool, datetime_str_format
from seyfert import VERSION
from seyfert.utils.shortcuts import ProbeName


class ConfigGUIWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super(ConfigGUIWindow, self).__init__()
        self.ui = Ui_SeyfertConfigGUI()
        self.ui.setupUi(self)
        self.ui.json_save_button.pressed.connect(partial(self.openDialog, file_format="JSON"))
        self.dialog = None
        self.outfile_line_edit = None
        self.close_btn = None
        self.args = args

    def openDialog(self, file_format: "str"):
        self.dialog = QDialog()
        self.dialog.resize(500, 250)
        self.dialog.setModal(True)
        self.dialog.setWindowTitle("Enter file name")
        self.outfile_line_edit = QtWidgets.QLineEdit(self.dialog)
        self.outfile_line_edit.setGeometry(QtCore.QRect(150, 100, 150, 30))
        self.close_btn = QtWidgets.QPushButton(self.dialog)
        self.close_btn.setGeometry(QtCore.QRect(400, 200, 100, 30))
        self.close_btn.setText("Save")
        if file_format == "JSON":
            self.close_btn.pressed.connect(self.dumpToJSONAndExit)
        else:
            raise ValueError(f"Invalid file format {file_format}, must be JSON")
        self.dialog.exec_()

    def dumpToJSONAndExit(self):
        gui_interface = GUIFileInterface(self.ui)
        json_data = gui_interface.dumpGUIToJSON()
        gui_interface.fc_json = JSONForecastConfig(data=json_data)

        filename = Path(self.outfile_line_edit.text()).name
        if not self.args.no_datetime:
            filename += f'_{VERSION}_{datetime_str_format(datetime.datetime.now())}'
        if '.json' not in Path(filename).suffixes:
            filename += '.json'
        gui_interface.fc_json.toJSON(filename)
        sys.exit(0)


class GUIFileInterface:
    PROBES_MAP = {
        'Lensing': 'Lensing',
        'GCph': 'PhotometricGalaxy',
        'GCsp': 'SpectroscopicGalaxy',
        'Void': 'Void'
    }
    INV_PROBES_MAP = {value: key for key, value in PROBES_MAP.items()}

    def __init__(self, ui: "Ui_SeyfertConfigGUI"):
        self.ui = ui
        self.xml_gen = None
        self.fc_json = None

    def dumpGUIToJSON(self) -> "Dict":
        json_data = {
            "metadata": {},
            "synthetic_opts": {},
            "survey": {
                "f_sky": self.readSurveyFSkyFromUI(),
                "shot_noise_file": None
            },
            "cosmology": dict(zip(["model", "flat", "parameters"], self.readCosmologyInfoFromUI())),
            "probes": self.buildJSONProbeOpts(),
            "derivative_settings": {
                "base_stem_disps": self.readSteMDispsFromUI()
            }
        }

        return json_data

    def readCosmoParsFromUI(self) -> "List[Dict]":
        params = []
        cosmo_table = self.ui.cosmo_pars
        for i in range(cosmo_table.rowCount()):
            row = tuple(cosmo_table.item(i, j).text() for j in range(cosmo_table.columnCount()))
            name = cosmo_table.verticalHeaderItem(i).text()
            params_data = {
                "name": name,
                "fiducial": float(row[0]),
                "current_value": float(row[0]),
                "kind": "CosmologicalParameter",
                "is_free_parameter": str_to_bool(row[1]),
                "units": row[2],
                "stem_factor": float(row[3]),
                "derivative_method": str(row[4])
            }
            params.append(params_data)

        return params

    def readSurveyFSkyFromUI(self) -> "float":
        return float(self.ui.f_sky.text())

    def readCosmologyInfoFromUI(self) -> "Tuple[str, bool, List[Dict]]":
        model = self.ui.cosmological_model.currentText()
        is_flat = bool(self.ui.is_universe_flat.checkState())
        cosmo_pars = self.readCosmoParsFromUI()

        return model, is_flat, cosmo_pars

    def readSteMDispsFromUI(self) -> "List[str]":
        stem_table = self.ui.stem_disps
        disps = [stem_table.item(i, 0).text() for i in range(stem_table.rowCount())]

        return disps

    def readProbesInfoFromUI(self) -> "Dict":
        main = self.ui.centralwidget
        stack_w = main.findChildren(QStackedWidget)[0]
        probes = list(filter(lambda x: isinstance(x, QWidget) and x.whatsThis() == 'probe', stack_w.children()))
        probe_opts = {
            ProbeName(p.objectName()).toAlias().name: {
                x.whatsThis(): x for x in p.findChildren(QWidget) if x.whatsThis()
            }
            for p in probes
        }

        return probe_opts

    def buildJSONProbeOpts(self) -> "Dict":
        probe_opts = self.readProbesInfoFromUI()

        return {
            name: self.buildGenericProbeJSONOpts(name=name, opts=probe_opts[name])
            for name in ["WL", "GCph", "GCsp", "V"]
        }

    def buildGenericProbeJSONOpts(self, name: "str", opts: "Dict") -> "Dict":
        probe_name = ProbeName(name)
        alias_name = probe_name.toAlias().name

        json_opts = {
            "long_name": probe_name.toLong().name,
            "presence_flag": bool(opts['presence_flag'].checkState())
        }

        ell_input_file = opts['ell_external_filename'].text()
        if ell_input_file:
            json_opts['ell_external_filename'] = ell_input_file
        else:
            json_opts.update({
                "l_min": opts['l_min'].value(),
                "l_max": opts['l_max'].value(),
                "ell_log_selection": bool(opts["ell_log_selection"].checkState())
            })
            if json_opts["ell_log_selection"]:
                json_opts["log_l_number"] = opts["log_l_number"].value()

        json_opts['density_file'] = opts['density_file'].text()

        if 'bias_file' in opts:
            json_opts["bias_file"] = opts['bias_file'].text()
        if 'bias_derivative_method' in opts:
            json_opts["bias_derivative_method"] = opts['bias_derivative_method'].currentText()
        if "marginalize_bias_flag" in opts:
            json_opts["marginalize_bias_flag"] = bool(opts["marginalize_bias_flag"].checkState())

        json_opts["specific_settings"] = self.buildProbeJSONSpecificSettings(name=alias_name, opts=opts)
        json_opts["extra_nuisance_parameters"] = self.buildProbeJSONExtraNuisanceParameters(probe_name=alias_name, opts=opts)

        return json_opts

    @staticmethod
    def buildProbeJSONSpecificSettings(name: "str", opts: "Dict") -> "Dict":
        spec_settings = {}
        if name == 'WL':
            spec_settings['include_IA'] = bool(opts['include_IA'].checkState())
        elif name == "V":
            spec_settings["void_kcut_invMpc"] = opts["void_kcut_invMpc"].value()
            spec_settings["void_kcut_width_invMpc"] = opts["void_kcut_width_invMpc"].value()
        elif name == "GCsp":
            spec_settings["compute_gcsp_cl_offdiag"] = bool(opts["compute_gcsp_cl_offdiag"].checkState())

        return spec_settings

    @staticmethod
    def buildProbeJSONExtraNuisanceParameters(probe_name: "str", opts: "Dict") -> "List":
        extra_nuis_params = []
        nuis_par_key_regex = re.compile(r'^nuis_param_([a-zA-Z0-9]+)$')
        if probe_name == 'WL':
            include_IA = bool(opts['include_IA'].checkState())
            for key, value in opts.items():
                match = nuis_par_key_regex.match(key)
                if match:
                    par_name = match.groups()[0]
                    fiducial = float(opts[key].text())
                    is_free_parameter = par_name != 'cIA' and include_IA
                    par_dict = dict(name=par_name, fiducial=fiducial, kind='NuisanceParameter', units='None',
                                    probe=probe_name, is_free_parameter=is_free_parameter)
                    extra_nuis_params.append(par_dict)

        return extra_nuis_params
