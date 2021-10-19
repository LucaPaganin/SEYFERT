from seyfert.utils import filesystem_utils as fsu
from seyfert.config.forecast_config import ForecastConfig

fc = ForecastConfig(input_file="/Users/lucapaganin/Downloads/test.json",
                    input_data_dir=fsu.default_seyfert_input_data_dir())

