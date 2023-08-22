# MVSEG_MW_2023
Contribution to the MVSEG Challenge (MICCAI 2023) by Marek Wodzinski (2nd place).

The challenge website: [Link](https://www.synapse.org/#!Synapse:syn51186045/wiki/)

Here you can see the full source code used to train / test the proposed solution.

Only the final experiment is left (the one used for the final Docker submission).

* In order to reproduce the experiment you should:
    * Download the MVSEG dataset [Link](https://www.synapse.org/#!Synapse:syn51186045/wiki/622044)
    * Update the [hpc_paths.py](./src/paths/hpc_paths.py) and [paths.py](./src/paths/paths.py) files.
    * Run the [parse_mvseg.py](./src/parsers/parse_mvseg.py)
    * Run the [run_aug_mvseg.py](./src/parsers/run_aug_mvseg.py)
    * Run the training using [run_segmentation_trainer.py](./src/runners/run_segmentation_trainer.py)
    * And finally use the trained model for inference using [inference.py](./src/inference/inference_mvseg.py)

The network was trained using HPC infrastructure (PLGRID). Therefore the .slurm scripts are omitted for clarity.

Please cite the MVSEG (TODO) if you found the source code useful.

Please find the method description: [Description](https://www.synapse.org/#!Synapse:syn52136026/wiki/623038)
