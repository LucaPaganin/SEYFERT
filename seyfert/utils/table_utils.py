from pathlib import Path
from typing import Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def write_dataframe_table(df: "pd.DataFrame", outfile: "Union[str, Path]", overwrite=False, **kwargs):
    outfile = Path(outfile)
    fmt = outfile.suffix
    if not outfile.exists() or (outfile.exists() and overwrite):
        if fmt == ".csv":
            df.to_csv(outfile, **kwargs)
        elif fmt == ".xlsx":
            df.to_excel(outfile, **kwargs)
        else:
            raise Exception(f"Unrecognized table file format {fmt}")
    else:
        logger.warning(f"file {outfile} already exists and overwrite is False, not overwriting it")
