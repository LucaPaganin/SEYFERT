import pytest
from seyfert.config import main_config as mcfg
from seyfert.utils import filesystem_utils as fsu

CFG_TEST_FILES = [fsu.config_files_dir() / x
                  for x in ["power_spectrum_config.json",
                            "angular_config.json",
                            "derivative_config.json",
                            "fisher_config.json"]]
TASKS = ["PowerSpectrum", "Angular", "Derivative", "Fisher"]


@pytest.mark.parametrize("task_name, json_input", zip(TASKS, CFG_TEST_FILES))
def test_config_for_task(json_input, task_name):
    cfg = mcfg.config_for_task(task_name, json_input=json_input)
    if task_name == "PowerSpectrum":
        assert isinstance(cfg, mcfg.PowerSpectrumConfig)
    elif task_name == "Angular":
        assert isinstance(cfg, mcfg.AngularConfig)
    elif task_name == "Derivative":
        assert isinstance(cfg, mcfg.DerivativeConfig)
    elif task_name == "Fisher":
        assert isinstance(cfg, mcfg.FisherConfig)
    else:
        raise TypeError(f'{type(cfg)} not an acceptable config for task {task_name}')


def test_fish_cfg_xc_derivatives_flags(cfg_fish_fixt):
    pass


