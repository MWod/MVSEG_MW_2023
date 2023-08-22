### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk

### Internal Imports ###
from paths import paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol




def parse_mvseg():
    input_path = p.raw_mvseg_path
    output_path = p.parsed_mvseg_path
    output_original_path = output_path / "Original"
    output_shape_path = output_path / "Shape_256_256_256"
    output_training_csv_path = output_path / "training_dataset.csv"
    output_validation_csv_path = output_path / "validation_dataset.csv"
    if not os.path.exists(output_original_path):
        os.makedirs(output_original_path)
    if not os.path.exists(output_shape_path):
        os.makedirs(output_shape_path)

    ### Parsing Params ###
    output_size = (256, 256, 256)
    device = "cuda:0"

    ### Parsing Training ###
    training_dataframe = []
    cases = os.listdir(os.path.join(input_path, "train"))
    input_cases = sorted([item for item in cases if "US" in item])
    input_gts = sorted([item for item in cases if "label" in item])

    for idx, current_case in enumerate(zip(input_cases, input_gts)):
        case, gt = current_case
        volume_path = os.path.join(input_path, "train", case)
        segmentation_path = os.path.join(input_path, "train", gt)
        print()
        print(f"Volume case: {volume_path}")
        print(f"Segmentation case: {segmentation_path}")

        volume, segmentation, volume_to_shape, segmentation_to_shape, spacing = parse_case(volume_path, segmentation_path, output_size, device)
        shape = volume.shape

        out_volume_path = os.path.join("train", str(idx), case)
        out_segmentation_path = os.path.join("train", str(idx), gt)
        training_dataframe.append((out_volume_path, out_segmentation_path))

        volume_to_shape_path = output_shape_path / out_volume_path
        segmentation_to_shape_path = output_shape_path / out_segmentation_path

        new_spacing = tuple(np.array(spacing) * np.array(shape) / np.array(output_size))
        print(f"Spacing: {spacing}")
        print(f"New Spacing: {new_spacing}")

        if not os.path.exists(os.path.dirname(volume_to_shape_path)):
            os.makedirs(os.path.dirname(volume_to_shape_path))

        to_save = sitk.GetImageFromArray(volume_to_shape.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(volume_to_shape_path))

        to_save = sitk.GetImageFromArray(segmentation_to_shape.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(segmentation_to_shape_path), useCompression=True)

    training_dataframe = pd.DataFrame(training_dataframe, columns=['Input Path', 'Ground-Truth Path'])
    training_dataframe.to_csv(output_training_csv_path, index=False)


    ### Parsing Validation ###
    val_dataframe = []
    cases = os.listdir(os.path.join(input_path, "val"))
    input_cases = sorted([item for item in cases if "US" in item])
    input_gts = sorted([item for item in cases if "label" in item])

    for idx, current_case in enumerate(zip(input_cases, input_gts)):
        case, gt = current_case
        volume_path = os.path.join(input_path, "val", case)
        segmentation_path = os.path.join(input_path, "val", gt)
        print()
        print(f"Volume case: {volume_path}")
        print(f"Segmentation case: {segmentation_path}")

        volume, segmentation, volume_to_shape, segmentation_to_shape, spacing = parse_case(volume_path, segmentation_path, output_size, device)
        shape = volume.shape

        out_volume_path = os.path.join("val", str(idx), case)
        out_segmentation_path = os.path.join("val", str(idx), gt)
        val_dataframe.append((out_volume_path, out_segmentation_path))

        volume_to_shape_path = output_shape_path / out_volume_path
        segmentation_to_shape_path = output_shape_path / out_segmentation_path

        new_spacing = tuple(np.array(spacing) * np.array(shape) / np.array(output_size))
        print(f"Spacing: {spacing}")
        print(f"New Spacing: {new_spacing}")

        if not os.path.exists(os.path.dirname(volume_to_shape_path)):
            os.makedirs(os.path.dirname(volume_to_shape_path))

        to_save = sitk.GetImageFromArray(volume_to_shape.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(volume_to_shape_path))

        to_save = sitk.GetImageFromArray(segmentation_to_shape.swapaxes(2, 1).swapaxes(1, 0))
        to_save.SetSpacing(new_spacing)
        sitk.WriteImage(to_save, str(segmentation_to_shape_path), useCompression=True)

    val_dataframe = pd.DataFrame(val_dataframe, columns=['Input Path', 'Ground-Truth Path'])
    val_dataframe.to_csv(output_validation_csv_path, index=False)




def parse_case(volume_path, segmentation_path, output_size, device):
    volume = sitk.ReadImage(volume_path)
    segmentation = sitk.ReadImage(segmentation_path)
    spacing = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
    segmentation = sitk.GetArrayFromImage(segmentation).swapaxes(0, 1).swapaxes(1, 2)
    print(f"Volume shape: {volume.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Spacing: {spacing}")

    volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    segmentation_tc = tc.from_numpy(segmentation.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Volume TC shape: {volume_tc.shape}")
    print(f"Segmentation TC shape: {segmentation_tc.shape}")

    resampled_volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
    resampled_segmentation_tc = pre_vol.resample_tensor(segmentation_tc, (1, 1, *output_size), mode='nearest')

    print(f"Resampled Volume TC shape: {resampled_volume_tc.shape}")
    print(f"Resampled Segmentation TC shape: {resampled_segmentation_tc.shape}")

    volume_tc = volume_tc[0, 0, :, :, :].detach().cpu().numpy()
    resampled_volume_tc = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy()

    segmentation_tc = segmentation_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)
    resampled_segmentation_tc = resampled_segmentation_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)

    return volume_tc, segmentation_tc, resampled_volume_tc, resampled_segmentation_tc, spacing

def run():
    parse_mvseg()

if __name__ == "__main__":
    run()