import os
import pathlib

data_path = None

### RAW Data Paths ###
raw_mvseg_path = data_path / 'MVSEG' / 'RAW'

### Parsed Data Paths ###
parsed_mvseg_path = data_path / "MVSEG" / "PARSED"

### Training Paths ###
project_path = None
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"