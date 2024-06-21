import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sys
import torch
import pymedphys
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from functools import partial

from let.cmd_args import inference_parser
from let.data import LETDatasetInMemory
from let.data_transform import get_preprocess_transforms
from let.model import (
    BasicUNetLETPredictor,
    FlexibleUNetLETPredictor,
    SegResNetLETPredictor,
    UNETRLETPredictor,
    collate_for_prediction,
)
from let import ntcp


def calculate_gamma(
    true_dose,
    predicted_dose,
    relevant_dose_area,
    dose_criteria,
    distance_criteria,
    gamma_setup,
):
    """
    Calculate the gamma passing rate between the true dose distribution and the predicted dose distribution.
    Assumes true_dose and predicted_dose to be interpolated to 1 x 1 x 1 mm^3 voxel spacing.

    Parameters
    ----------
    true_dose : numpy.ndarray
        The true dose distribution.
    predicted_dose : numpy.ndarray
        The predicted dose distribution.
    relevant_dose_area : numpy.ndarray
            A boolean array of the same shape as true_dose, where True values indicate the voxels that are relevant for the gamma calculation.
    dose_criteria : float
        The dose difference criteria, in percentage of the local dose.
    distance_criteria : float
        The distance-to-agreement criteria, in mm.
    gamma_setup: string
        Sets global or local gamma analysis

    Returns
    -------
    gamma : numpy.ndarray
        The gamma index map. Same shape as true_dose or predicted_dose.

    """

    assert (
        true_dose.shape == predicted_dose.shape
    ), "True and predicted dose maps must have the same shape."

    # gets the reference coordinates. Expects true_dose to have 1mm voxel
    coords = (
        np.arange(np.shape(true_dose)[0]),
        np.arange(np.shape(true_dose)[1]),
        np.arange(np.shape(true_dose)[2]),
    )

    if gamma_setup == "local":
        local_gamma = True
    else:
        local_gamma = False

    # set regions without relevant dose to 0 to definetly exclude them in the evaluation
    true_dose[~relevant_dose_area] = 0
    predicted_dose[~relevant_dose_area] = 0

    max_dose = np.max(true_dose[relevant_dose_area])

    print("Perform gamma analysis....")
    gamma = pymedphys.gamma(
        axes_reference=coords,
        dose_reference=true_dose,
        axes_evaluation=coords,
        dose_evaluation=predicted_dose,
        dose_percent_threshold=dose_criteria,
        distance_mm_threshold=distance_criteria,
        global_normalisation=max_dose,
        # calc gamma index for all voxels above the minimum value within the relevant dose area to speed it up
        lower_percent_dose_cutoff=10,
        # sets the dose_criteria to be normed on global max dose
        local_gamma=local_gamma,
        # just calculates to max gamma of 1.1 to speed up calculations
        max_gamma=1.1,
        interp_fraction=5,
    )
    print("Gamma_indexes calculated!")

    return gamma


def evaluate_gamma(gamma_map, true_dose_map, roi_mask, relevant_dose_area):
    """
    Evaluate the gamma index passing rates using dose bins and ROI, and return the results in a pandas DataFrame.

    Parameters
    ----------
    gamma_map : numpy.ndarray
        The gamma index map, represented as a 3D numpy array.
    true_dose_map : numpy.ndarray
        The true dose distribution map, represented as a 3D numpy array.
    roi_mask : numpy.ndarray
        A boolean mask representing the region of interest.
    relevant_dose_area: numpy.ndarray
        A boolean mask representing the region of relevant dose

    Returns
    -------
    results_df : pandas.DataFrame
        A DataFrame containing the gamma index passing rates and number of passed voxels
        for each dose bin and for complete dose range. Dose ranges are normed on max dose of
        true_dose map. If roi does not have voxels of specific dose range, nan values gets inserted.
    """

    assert (
        gamma_map.shape == true_dose_map.shape
    ), "Gamma map and true dose map must have the same shape."

    if roi_mask is not None:
        assert (
            roi_mask.shape == true_dose_map.shape
        ), "ROI mask must have the same shape as the true dose map."

    # Create empty lists to store the passing rate, number of voxels and column names for each dose bin
    result = {}

    # As gamma analysis might be not calculated for whole image, handle nan values correctly
    gamma_map_result = np.where(np.isnan(gamma_map), np.nan, gamma_map < 1)
    passing_rate = np.nanmean(gamma_map_result[roi_mask])
    num_passed = np.nansum(gamma_map_result[roi_mask])

    # Add the passing rate, number of passed voxels and column names to the result dict
    result["Passing rate"] = passing_rate
    result["Number of passed voxels"] = num_passed

    print(f"Passing rate: {passing_rate} | Number of passed voxels: {num_passed}")

    return result


def compute_xcc_metric(data_arr, x):
    """
    Computes the minimum among the (x * 1000) voxels with the largest values.
    Assumes data_tensor to be interpolated to 1 x 1 x 1 mm^3 voxel spacing

    Parameters
    ----------
    data_arr: 1d numpy array of floats
    x: float
        fraction/multiple of cubic centimeters
    """

    n_voxels = int(x * 1000)  # 1000 = 10^3 = number of voxels in one cm^3
    # TODO: what if n_voxels > number of elements in data_arr?
    if len(data_arr) < n_voxels:
        print(
            f"[WW]: in compute_{x}cc: n_voxels={n_voxels}, but input has "
            f"only length {len(data_arr)}!"
        )

    # get the minimum of the n_voxels largest values
    # which is kind of a percentile, but we fix the number of voxels, not
    # the fraction of total location

    # from largest to smallest
    data_sorted = np.sort(data_arr.flatten())[::-1]

    # take the minimum of the n_voxels largest values
    # which should be the same as data_sorted[n_voxels-1]
    # this does not fail if n_voxels > number of elements in data_sorted
    # and will take the min over all available elements then. This might
    # bias the statistics!

    # return data_sorted[n_voxels-1]  # this fails if n_voxels > len(data_arr)
    return data_sorted[:n_voxels].min()


def ipsi_and_contralateral_lookup(plan_data_dict, rois_to_evaluate):
    """
    For regions of interest that occur on the left and right hemisphere
    of a patient plan, determine which of them is ipsilateral and which
    is contralateral. We define the ipsilateral side to be the one with
    the higher mean dose. ROIs containing the 'relevant_dose_area_' tag
    will be skipped as for those ROIs, the value from the corresponding
    ROI should be re-used.

    Parameters
    ----------
    plan_data_dict: dict
        all rois are required to start with Mask_<ROI> and should be keys, while
        values should be boolean np.arrays
    rois_to_evaluate: list of str
        names of the ROIs to consider. It is expected that if the name is <ROI>,
        then a key Mask_<ROI> is present in plan_data_dict

    Returns
    -------
    Dictionary where keys are ROI names and values are "ipsi", "contra"
    or np.nan for ROI names not ending with "_L" or "_R"
    """
    # determine the names of the ROIs that we need to consider (but without
    # the potentially occuring _R and _L suffix)

    roi_names_unpaired = []
    roi_names_paired = []
    for roi in rois_to_evaluate:
        if "relevant_dose_area_" in roi:
            # we skip ROIS containing the "relevant_dose_area_" because for such
            # ROIs, we re-use the ipsi/contra value from the ROI obtained without
            # taking into account the relevant_dose_area
            continue
        if roi.endswith("_L") or roi.endswith("_R"):
            roi_names_paired.append(roi[:-2])
        else:
            # NOTE: also the "relevant_dose_area" without the trailing underscore
            roi_names_unpaired.append(roi)
    # drop the double entries that occur by looping over e.g. Lens_L and Lens_R
    roi_names_paired = list(set(roi_names_paired))

    # ipsi/contra does not make sense for unpaired ROIs, so this will be nan
    ipsicontra_mapping = {r: np.nan for r in roi_names_unpaired}

    for roi_base in roi_names_paired:
        roiname_l = f"Mask_{roi_base + '_L'}"
        roiname_r = f"Mask_{roi_base + '_R'}"
        assert roiname_l in plan_data_dict
        assert roiname_r in plan_data_dict

        roi_l = plan_data_dict[roiname_l]
        roi_r = plan_data_dict[roiname_r]

        # get dose for left and right for comparison
        dose = plan_data_dict["dose"]
        l_dose_mean = dose[roi_l].mean()
        r_dose_mean = dose[roi_r].mean()

        if l_dose_mean >= r_dose_mean:
            ipsicontra_mapping[roi_base + "_L"] = "ipsi"
            ipsicontra_mapping[roi_base + "_R"] = "contra"
        else:
            ipsicontra_mapping[roi_base + "_L"] = "contra"
            ipsicontra_mapping[roi_base + "_R"] = "ipsi"

    return ipsicontra_mapping


def evaluate_sample(
    plan_data_dict, voxel_aggregations, rois_to_evaluate, voxel_error_types
):
    """
    Parameters
    ----------
    plan_data_dict: dict with info and data for a given plan together with 'let_prediction' key
                    mapping to a 3D numpy array. Also rois should be present, indicated by
                    keys starting with 'Mask_<roi_name>', where <roi_name> should be present
                    in the 'rois_to_evaluate' parameter

    Returns
    -------
    A pandas DataFrame with len(rois_to_evaluate) * len(voxel_aggregations)
    rows per plan
    and len(voxel_error_types) + 2 metrics for ground truth, prediction
    and error scores.
    NOTE: additional columns are for plan_id, roi, voxel_aggregation.
    """
    # lookup dict for the aggregation functions
    # added the L_x-cc aggregations from this paper: https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.16043
    # for x = 0.1, 0.3, 1
    voxel_aggregation_fns = {
        "median": np.median,
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "1_percentile": partial(np.percentile, q=1),
        "2_percentile": partial(np.percentile, q=2),
        "98_percentile": partial(np.percentile, q=98),
        "99_percentile": partial(np.percentile, q=99),
    }

    # lookup dict for the error types
    voxel_error_type_fns = {
        "unsigned_absolute": lambda gt, pred: np.abs(gt - pred),
        "signed_absolute": lambda gt, pred: gt - pred,
    }

    sample_plan = plan_data_dict["plan_id"]
    sample_gt = plan_data_dict["let"]
    sample_pred = plan_data_dict["let_prediction"]

    # one line in the dataframe for each roi of a patient
    # None means the whole volume irrespective of a ROI
    results = []

    ipsicontra_map = ipsi_and_contralateral_lookup(plan_data_dict, rois_to_evaluate)
    # translate the value to what will be written in the csv file in the
    # "roi_is_ipsilateral" column
    ipsicontra_val_map = {np.nan: np.nan, "ipsi": True, "contra": False}

    for roi in [None] + rois_to_evaluate:
        if roi is None:
            sample_roi = np.ones_like(sample_gt).astype(bool)
        elif f"Mask_{roi}" in plan_data_dict:
            sample_roi = plan_data_dict[f"Mask_{roi}"]
        else:
            # the ROI is not present for the current patient
            print(f"[WW]: {sample_plan} has no ROI {roi}. Will skip!")
            continue

        if np.sum(sample_roi) == 0:
            print(f"[WW]: {sample_plan} ROI {roi} is empty. Will skip!")
            continue

        sample_gt_roi = sample_gt[sample_roi]
        sample_pred_roi = sample_pred[sample_roi]

        if roi is None:
            sample_roi_is_ipsi = np.nan
        elif roi.startswith("relevant_dose_area_"):
            # we should re-use the ipsi/contra definition from the ROI which did
            # not consider the relevant dose area
            # the matching name of the ROI without the relevant dose area
            roi_key = roi.split("relevant_dose_area_")[1]
            ipsicontra = ipsicontra_map[roi_key]

            sample_roi_is_ipsi = ipsicontra_val_map[ipsicontra]
        else:
            ipsicontra = ipsicontra_map[roi]
            sample_roi_is_ipsi = ipsicontra_val_map[ipsicontra]

        # we can summarize the voxel values in the ROI in different
        # ways (mean, median, max...)
        for voxel_aggregation in voxel_aggregations:
            agg_fn = voxel_aggregation_fns[voxel_aggregation]
            # compute aggregation for gt in roi
            # compute aggregation for pred in roi
            # compute aggregation on error in roi
            sample_roi_result = {
                "plan_id": sample_plan,
                "roi": roi,
                "roi_is_ipsilateral": sample_roi_is_ipsi,
                "voxel_aggregation": voxel_aggregation,
                "gt": agg_fn(sample_gt_roi),
                "pred": agg_fn(sample_pred_roi),
            }

            # this adds one column for each voxel error type as we can
            # determine the error in different ways before aggregating
            # by the voxels
            for voxel_error_type in voxel_error_types:
                err_f = voxel_error_type_fns[voxel_error_type]
                # aggregation of voxelwise errors, e.g.
                # mean(gt - pred) or mean(|gt - pred|)
                sample_errors_roi = agg_fn(err_f(sample_gt_roi, sample_pred_roi))
                # error between voxel aggregations, called distributionwise error, e.g.
                # mean(gt) - mean(pred) or |mean(gt) - mean(pred)|
                dist_errors_roi = err_f(agg_fn(sample_gt_roi), agg_fn(sample_pred_roi))

                sample_roi_result[f"voxelwise_{voxel_error_type}_error"] = (
                    sample_errors_roi
                )
                sample_roi_result[f"distributionwise_{voxel_error_type}_error"] = (
                    dist_errors_roi
                )

            results.append(sample_roi_result)

        # xcc metrics come here as they can't be computed voxelwise
        # NOTE: this assumes input to be interpolated to 1mm^3evaluate_sample_gamma
        # this can't be computed voxelwise so those entries should stay with NaN then
        for x in [0.1, 0.3, 1]:
            gt = compute_xcc_metric(sample_gt_roi, x=x)
            pred = compute_xcc_metric(sample_pred_roi, x=x)

            item = {
                "plan_id": sample_plan,
                "roi": roi,
                "roi_is_ipsilateral": sample_roi_is_ipsi,
                "voxel_aggregation": f"{x}_cc",
                "gt": gt,
                "pred": pred,
            }
            for voxel_error_type in voxel_error_types:
                err_f = voxel_error_type_fns[voxel_error_type]
                item[f"distributionwise_{voxel_error_type}_error"] = err_f(
                    item["gt"], item["pred"]
                )

            results.append(item)

    return pd.DataFrame(results)


def evaluate_sample_gamma(
    plan_data_dict,
    rois_to_evaluate,
    gamma_configuration,
    clinical_var_df_patient,
    dose_is_rbe_weighted,
):
    sample_plan = plan_data_dict["plan_id"]
    sample_gt = plan_data_dict["let"]
    sample_pred = plan_data_dict["let_prediction"]
    sample_dose = plan_data_dict["dose"]

    relevant_dose_area = plan_data_dict["Mask_relevant_dose_area"]

    results = []

    for config in gamma_configuration:
        input_modality = config[0]
        assert input_modality in [
            "LET",
            "wedenberg",
            "bahn",
            "constant",
            "dose*LET",
        ], f"{input_modality} not implemented in gamma analysis or as varRBE model"

        gamma_setup = config[1]
        assert gamma_setup in [
            "local",
            "global",
        ], f"{gamma_setup} must be 'global' or 'local'"

        dose_criteria = config[2]
        assert dose_criteria >= 0, "Dose criteria must be non-negative."

        distance_criteria = config[3]
        assert distance_criteria >= 0, "Distance criteria must be non-negative."

        if input_modality == "LET":
            input_gamma_true = sample_gt
            input_gamma_pred = sample_pred
        else:
            let_to_rbe_converter_normal_tissue = ntcp.LETtoRBEConverter(
                input_modality, alphabeta=2.0, rbe_constant=1.1
            )

            input_gamma_true = let_to_rbe_converter_normal_tissue(
                dose=sample_dose,
                let=sample_gt,
                dose_is_rbe_weighted=dose_is_rbe_weighted,
                clinical_var_df=clinical_var_df_patient,
            )
            input_gamma_pred = let_to_rbe_converter_normal_tissue(
                dose=sample_dose,
                let=sample_pred,
                dose_is_rbe_weighted=dose_is_rbe_weighted,
                clinical_var_df=clinical_var_df_patient,
            )

            # Overwrite CTV region with alphabeta = 10
            let_to_rbe_converter_CTV = ntcp.LETtoRBEConverter(
                input_modality, alphabeta=10.0, rbe_constant=1.1
            )

            input_gamma_true[plan_data_dict["Mask_CTV"]] = let_to_rbe_converter_CTV(
                dose=sample_dose,
                let=sample_gt,
                dose_is_rbe_weighted=dose_is_rbe_weighted,
                clinical_var_df=clinical_var_df_patient,
            )[plan_data_dict["Mask_CTV"]]

            input_gamma_pred[plan_data_dict["Mask_CTV"]] = let_to_rbe_converter_CTV(
                dose=sample_dose,
                let=sample_pred,
                dose_is_rbe_weighted=dose_is_rbe_weighted,
                clinical_var_df=clinical_var_df_patient,
            )[plan_data_dict["Mask_CTV"]]

        print(
            f"Calculating {gamma_setup} gamma map for {input_modality} with {dose_criteria} dose difference and {distance_criteria} distance-to-agreement"
        )
        sample_gamma = calculate_gamma(
            true_dose=input_gamma_true,
            predicted_dose=input_gamma_pred,
            dose_criteria=dose_criteria,
            distance_criteria=distance_criteria,
            relevant_dose_area=relevant_dose_area,
            gamma_setup=gamma_setup,
        )
        print("Gamma map ready!")

        # one line in the dataframe for each roi of a plan
        for roi in rois_to_evaluate:
            if f"Mask_{roi}" in plan_data_dict:
                sample_roi = plan_data_dict[f"Mask_{roi}"]
            else:
                # the ROI is not present for the current plan
                print(f"[WW]: {sample_plan} has no ROI {roi}. Will skip!")
                continue

            print(f"ROI: {roi}")
            # gamma analysis evaluation
            gamma_results = evaluate_gamma(
                sample_gamma,
                sample_gt,
                roi_mask=sample_roi,
                relevant_dose_area=relevant_dose_area,
            )
            gamma_results["plan_id"] = sample_plan
            gamma_results["modality"] = input_modality
            gamma_results["gamma setup"] = gamma_setup
            gamma_results["roi"] = roi
            gamma_results["dose difference"] = dose_criteria
            gamma_results["distance-to-agreement"] = distance_criteria

            results.append(gamma_results)

    return pd.DataFrame(results)


def generate_masks(
    sample_gt,
    sample_dose,
    sample_ctv_mask,
    dose_rel,
    let_bins_abs,
    dose_bins_abs,
    clip_let_below_dose,
):
    """
    Generate masks based on different binning strategies and thresholds.

    :param sample_gt: Ground truth samples.
    :param sample_dose: Dose samples.
    :param sample_ctv_mask: Mask of ctv.
    :param dose_rel: Percentiles to consider for the to the mean dose in
    the CTV relative dose bins.
    :param let_bins_abs: Bin edges to consider for the absolute LET bins.
    :param dose_bins_abs: Bin edges to consider for the absolute dose bins.
    :return: A dictionary containing masks for each bin.
    """

    result = {}
    mask_names = []

    # Initialize bins
    dict_bins = {"dose": {"abs": dose_bins_abs}, "let": {"abs": let_bins_abs}}

    # Generate masks for each bin
    for dist_type in dict_bins.keys():  # dose/let
        for bin_type in dict_bins[dist_type].keys():  # abs/rel
            bins = dict_bins[dist_type][bin_type]

            bin_dist = sample_gt if dist_type == "let" else sample_dose
            percentiles = dose_rel

            for i in range(len(bins)):
                if i == len(bins) - 1:
                    mask = bin_dist > bins[i]
                    label1 = str(bins[i]) if bin_type == "abs" else str(percentiles[i])
                    label2 = "max"
                else:
                    mask = (bin_dist > bins[i]) & (bin_dist <= bins[i + 1])
                    label1 = str(bins[i]) if bin_type == "abs" else str(percentiles[i])
                    label2 = (
                        str(bins[i + 1])
                        if bin_type == "abs"
                        else str(percentiles[i + 1])
                    )

                key = f"Mask_{dist_type.capitalize()}_between_{label1}_and_{label2}_{bin_type.capitalize()}"
                result[key] = mask.astype(bool)
                mask_names.append(key[5:])

    key = "Mask_relevant_dose_area"
    if clip_let_below_dose is not None:
        # generate mask for relevant dose are
        mask = sample_dose > clip_let_below_dose
    else:
        mask = np.ones(np.shape(sample_dose))
    result[key] = mask.astype(bool)
    mask_names.append(key[5:])

    return result, mask_names


def compute_metrics(
    dataset,
    plan_predictions,
    rois_to_evaluate,
    voxel_aggregations,
    voxel_error_types,
    dose_rel,
    let_bins_abs,
    dose_bins_abs,
    clip_let_below_dose,
):
    """
    Function to compute metrics based on step outputs.

    :param dataset: LETDatasetInMemory instance
    :param plan_predictions: dict from plan_ids to numpy arrays of LET predictions
    :param rois_to_evaluate: List of regions of interest to evaluate.
    :param voxel_aggregations: Aggregations for voxel analysis.
    :param voxel_error_types: Error types for voxel analysis.
    :param let_percentiles: Percentiles to consider for the relative LET bins.
    :param dose_percentiles: Percentiles to consider for the relative dose bins.
    :param let_bins_abs: Bin edges to consider for the absolute LET bins.
    :param dose_bins_abs: Bin edges to consider for the absolute dose bins.
    :param clip_let_below_dose: (normalized) dose level from which
                                      upward we consider the relevant dose area
    :return: List of batch metrics.
    """
    metrics = []
    for plan_idx in tqdm(range(len(dataset)), desc="Computing metrics"):
        plan_data_dict = dataset[plan_idx]
        plan_id = plan_data_dict["plan_id"]

        eval_data = {
            "plan_id": plan_data_dict["plan_id"],
            # convert to numpy, discard color channel
            "dose": plan_data_dict["dose"].numpy()[0],
            # convert to numpy, discard color channel
            "let": plan_data_dict["let"].numpy()[0],
            # discard color channel, is already numpy
            "let_prediction": plan_predictions[plan_id][0],
        }

        # copy all masks (they dont have color channel)
        assert "Mask_CTV" in plan_data_dict, "No CTV in available!"
        for k in plan_data_dict:
            if k.startswith("Mask_"):
                eval_data[k] = plan_data_dict[k]

        # generate further masks based on the dose and LET statistics
        # e.g. regions where dose/LET is above a given value
        # NOTE: here the 'relevant_dose_area' mask is also created
        masks_dict, mask_names = generate_masks(
            eval_data["let"],
            eval_data["dose"],
            eval_data["Mask_CTV"],
            dose_rel,
            let_bins_abs,
            dose_bins_abs,
            clip_let_below_dose,
        )

        # Insert masks into the plan_data_dict
        for key in masks_dict:
            eval_data[key] = masks_dict[key]

        # add the masks of ROI * "Mask_relevant_dose_area"
        # to the eval_data (with Mask_ prefix) and to the mask_names
        # (without the Mask_ prefix)
        relevant_dose_roi = eval_data["Mask_relevant_dose_area"]
        rois_and_relevant_dose = []
        for k in plan_data_dict:
            if k.startswith("Mask_") and k != "Mask_relevant_dose_area":
                roi = plan_data_dict[k]
                roi_name = k.split("Mask_")[1]
                roi_name_new = f"relevant_dose_area_{roi_name}"

                eval_data[f"Mask_{roi_name_new}"] = roi & relevant_dose_roi
                rois_and_relevant_dose.append(roi_name_new)

        rois = rois_to_evaluate + mask_names + rois_and_relevant_dose
        metrics.append(
            evaluate_sample(
                eval_data,
                voxel_aggregations=voxel_aggregations,
                rois_to_evaluate=rois,
                voxel_error_types=voxel_error_types,
            )
        )

    return pd.concat(metrics, ignore_index=True)


def compute_gamma(
    dataset,
    plan_predictions,
    rois_to_evaluate,
    gamma_configuration,
    plan_table_path,
    dose_rel,
    let_bins_abs,
    dose_bins_abs,
    clip_let_below_dose,
    dose_is_rbe_weighted,
    use_multithreading,
):
    metrics = []

    if plan_table_path is not None:
        # Read all sheets from the Excel file into a dictionary of DataFrames
        clinical_var_df_dict = pd.read_excel(plan_table_path, sheet_name=None)
        # Concatenate all DataFrames into a single DataFrame
        clinical_var_df_combined = pd.concat(
            clinical_var_df_dict.values(), ignore_index=True
        )
    else:
        clinical_var_df_combined = None

    def process_plan(plan_idx):
        plan_data_dict = dataset[plan_idx]
        plan_id = plan_data_dict["plan_id"]

        eval_data = {
            "plan_id": plan_data_dict["plan_id"],
            # convert to numpy, discard color channel
            "dose": plan_data_dict["dose"].numpy()[0],
            # convert to numpy, discard color channel
            "let": plan_data_dict["let"].numpy()[0],
            # discard color channel, is already numpy
            "let_prediction": plan_predictions[plan_id][0],
        }

        # copy all masks (they dont have color channel)
        assert "Mask_CTV" in plan_data_dict, "No CTV in available!"
        for k in plan_data_dict:
            if k.startswith("Mask_"):
                eval_data[k] = plan_data_dict[k]

        # generate further masks based on the dose and LET statistics
        # e.g. regions where dose/LET is above a given value
        # NOTE: here the 'relevant_dose_area' mask is also created
        masks_dict, mask_names = generate_masks(
            eval_data["let"],
            eval_data["dose"],
            eval_data["Mask_CTV"],
            dose_rel,
            let_bins_abs,
            dose_bins_abs,
            clip_let_below_dose,
        )

        # Insert masks into the plan_data_dict
        for key in masks_dict:
            eval_data[key] = masks_dict[key]

        # add the masks of ROI * "Mask_relevant_dose_area"
        # to the eval_data (with Mask_ prefix) and to the mask_names
        # (without the Mask_ prefix)
        relevant_dose_roi = eval_data["Mask_relevant_dose_area"]
        rois_to_compute_gamma = []
        rois_to_compute_gamma.append("relevant_dose_area")

        for k in list(eval_data.keys()):
            if k.startswith("Mask_") and k != "Mask_relevant_dose_area":
                roi = eval_data[k]
                roi_name = k.split("Mask_")[1]
                roi_name_new = f"relevant_dose_area_{roi_name}"

                eval_data[f"Mask_{roi_name_new}"] = roi & relevant_dose_roi
                rois_to_compute_gamma.append(roi_name_new)

        if clinical_var_df_combined is not None:
            # Filter the combined DataFrame based on the plan_id
            clinical_var_df_patient = clinical_var_df_combined.loc[
                clinical_var_df_combined["plan_id"] == eval_data["plan_id"], :
            ]
        else:
            clinical_var_df_patient = None

        return evaluate_sample_gamma(
            eval_data,
            rois_to_evaluate=rois_to_compute_gamma,
            gamma_configuration=gamma_configuration,
            clinical_var_df_patient=clinical_var_df_patient,
            dose_is_rbe_weighted=dose_is_rbe_weighted,
        )

    if use_multithreading:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_plan, plan_idx)
                for plan_idx in range(len(dataset))
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Computing gamma passing rates",
            ):
                metrics.append(future.result())
    else:
        for plan_idx in tqdm(range(len(dataset)), desc="Computing gamma passing rates"):
            metrics.append(process_plan(plan_idx))

    return pd.concat(metrics, ignore_index=True)


def main(args):
    # no seeding is necessary since inference does not involve randomness
    plan_ids = pd.read_csv(args.valid_id_file, header=None).values.squeeze().tolist()

    dataset = LETDatasetInMemory(
        data_dir=args.data_dir,
        plan_ids=plan_ids,
        ct_filename=args.ct_filename if args.use_ct else None,
        dose_filename=args.dose_filename,
        let_filename=args.let_filename,
        roi_filename=args.roi_filename,
        crop_size=args.crop_size,
        return_rois=args.rois_to_evaluate,
        preprocess_transform=get_preprocess_transforms(),
        augmentation_transform=None,
        clip_let_below_dose=args.clip_let_below_dose,
        multiply_let_by_dose=args.multiply_let_by_dose,
        ct_window=args.ct_window,
        ct_normalisation=not args.no_ct_normalisation,
    )

    # custom collate function to avoid putting
    # all the loaded ROIs to the device!
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_for_prediction,
    )

    if args.model_type == "basic_unet":
        model_cls = BasicUNetLETPredictor
    elif args.model_type == "flex_unet":
        model_cls = FlexibleUNetLETPredictor
    elif args.model_type == "segresnet":
        model_cls = SegResNetLETPredictor
    elif args.model_type == "unetr":
        model_cls = UNETRLETPredictor

    model = model_cls.load_from_checkpoint(args.ckpt_file)
    model.eval()
    model.freeze()
    print(f"Loaded trained model from checkpoint {args.ckpt_file}.")

    # get the predictions only, not the masks
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=args.gpus
    )
    batch_predictions = trainer.predict(model, loader)

    # get the predictions as a dictionary for each plan, not batched anymore
    plan_predictions = {}
    for batch_idx, batch_dict in enumerate(loader):
        plan_ids = batch_dict["plan_id"]
        batch_preds = batch_predictions[batch_idx]
        assert len(plan_ids) == len(batch_preds)

        for pidx in range(len(plan_ids)):
            plan_id = plan_ids[pidx]
            plan_predictions[plan_id] = batch_preds[pidx].detach().cpu().numpy()

    output_dir = Path(args.output_dir)

    if args.execute_metric_computation:
        metric_df = compute_metrics(
            dataset,
            plan_predictions,
            rois_to_evaluate=args.rois_to_evaluate,
            voxel_aggregations=args.voxel_aggregation,
            voxel_error_types=args.voxel_error_type,
            dose_rel=np.array(args.dose_rel),
            let_bins_abs=np.array(args.let_bins_abs),
            dose_bins_abs=np.array(args.dose_bins_abs),
            clip_let_below_dose=args.clip_let_below_dose,
        )

        print(metric_df)
        print(metric_df.describe())
        metric_df.to_csv(output_dir / "model_performance.csv", index=False)
    if not args.no_gamma:
        gamma_df = compute_gamma(
            dataset,
            plan_predictions,
            rois_to_evaluate=args.rois_to_evaluate,
            gamma_configuration=args.gamma_configuration,
            plan_table_path=args.plan_table_path,
            dose_rel=np.array(args.dose_rel),
            let_bins_abs=np.array(args.let_bins_abs),
            dose_bins_abs=np.array(args.dose_bins_abs),
            clip_let_below_dose=args.clip_let_below_dose,
            dose_is_rbe_weighted=not args.physical_dose,
            use_multithreading=args.use_gamma_multithreading,
        )

        print(gamma_df)
        print(gamma_df.describe())
        gamma_df.to_csv(output_dir / "model_performance_gamma.csv", index=False)
    else:
        print("No gamma-analysis will be performed.")

    return 0


if __name__ == "__main__":
    parser = inference_parser("LET model inference")

    args = parser.parse_args()
    print("Model Inference")
    print("\nParsed args are\n")
    pprint(args)

    output_dir = args.output_dir
    if output_dir is None:
        exp_name = args.model_type
        output_dir = f"./experiments/{exp_name}/inference"
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    else:
        raise ValueError(f"output_dir {output_dir} already exists!")

    print(f"\nUsing {output_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(output_dir / "commandline_args.json", "w") as of:
        json.dump(vars(args), of, indent=2)

    retval = main(args)
    sys.exit(retval)
