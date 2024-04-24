import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from skimage.measure import regionprops
from tqdm.auto import tqdm

from let.data_transform import get_preprocess_transforms


def fuzzy_find_name(name_list, search_key, only_exact=False):
    """
    Tries to find names in a dictionary that match search_key.
    If only_exact is False, also looks for case-insensitive
    matches as a first fallback. A second fallback looks for
    case-insensitive matches that start with the search_key.

    Returns
    -------
    None if the search was not successful.
    A list of fuzzy matches otherwise.
    """
    if search_key in name_list:
        return [search_key]

    if only_exact:
        # we stop the search here if we only want to retrieve
        # exact matches
        return None

    # we did not find the exact key in the dictionary
    # so we try looking for lower case matches
    for k in name_list:
        if k.lower() == search_key.lower():
            print(f"[W]: Did not find {search_key} exactly, "
                  f"but {k} matches case-insensitive.")
            return [k]

    # check if we can find keys that start with
    # the provided search key
    startwith_matches = []
    for k in name_list:
        if k.lower().startswith(search_key.lower()):
            startwith_matches.append(k)

    if len(startwith_matches) > 0:
        print(f"[W]: Did not find {search_key} exactly, "
              f"but {startwith_matches} start similarly "
              f"(case-insensitive.)")
        return startwith_matches

    # we give up
    return None


def fuse_rois(list_of_rois):
    """
    Parameters
    ----------
    list_of_rois: list of np.array of type bool

    Returns
    -------
    the union of all rois
    """
    return np.logical_or.reduce(list_of_rois)


def crop_with_padding3d(img, center, crop_size):
    """
    Crop a given image to a given size centered at given coordinates.
    Applies padding with minimum value if crop_size exceeds the
    image dimensions.

    Parameters
    ----------
    img: 3D np.array
    center: tuple/list of length 3
        contain the center coordinates of the crop
        in z, y and x dimension
    crop_size: tuple/list of length 3
        contain the size of the desired crop
        in z, y and x dimension
    """
    img_size = np.array(img.shape)
    center = np.array(center)
    crop_size = np.array(crop_size)

    assert np.all(center < img_size)
    assert len(img_size) == len(center) == len(crop_size) == 3

    crop_size_half = crop_size // 2

    # starting and ending indices of the crop
    mins = center - crop_size_half
    maxs = mins + crop_size

    # determine padding depending on the computed starting/ending indices
    # 0 if the starting positions are >= 0, -mins otherwise => pad(x) = -min(x, 0)
    pad_lower = -np.minimum(mins, 0)
    # 0 if the maxs are <= shape, maxs - shape otherwise => pad(x) = max(x - shape, 0)
    pad_upper = np.maximum(maxs - img_size, 0)

    # not padding with zeros, but with minimal value
    img_padded = np.pad(
        img, tuple(zip(pad_lower, pad_upper)),
        mode="constant",
        constant_values=img.min())
    # now the center of the original image expressed in coordinates of the padded image
    center_in_padded = center + pad_lower
    lower_in_padded = center_in_padded - crop_size_half
    upper_in_padded = lower_in_padded + crop_size

    padded_crop = img_padded[
        lower_in_padded[0]:upper_in_padded[0],
        lower_in_padded[1]:upper_in_padded[1],
        lower_in_padded[2]:upper_in_padded[2]]

    assert np.all(np.array(padded_crop.shape) == crop_size)
    return padded_crop


def create_crop_based_on_external_contour(img_dict, external_roi, brainstem_roi, crop_size):
    """
    Parameters
    ----------
    img_dict: dict
        keys are arbitrary and values should be 3d numpy arrays (like for dose, let, ...)
    external_roi: 3d np.array
        The binary mask containing the 'External' contour
    brainstem_roi: 3d np.array
        The binary mask containing the brainstem contour
    crop_size: tuple/list of length 3
        Determines the size all dictionary entries will be cropped to, using zero
        padding where necessary.
    """
    external_roi = external_roi.astype(np.uint8)
    brainstem_roi = brainstem_roi.astype(np.uint8)

    # determine bounding box of brainstem
    props_brainstem = regionprops(brainstem_roi)
    assert len(props_brainstem) == 1
    props_brainstem = props_brainstem[0]

    brainstem_bbox = props_brainstem.bbox
    z_low = brainstem_bbox[0]

    # disregard the parts of external below the brainstem
    external_roi[:z_low] = 0

    # now compute extent of the clipped external contour
    props_external = regionprops(external_roi)
    assert len(props_external) == 1
    props_external = props_external[0]

    min_z, min_y, min_x, max_z, max_y, max_x = props_external.bbox

    # compute the center coordinate of the crop as the average coordinate of the bounding box
    center_z = int(.5 * (min_z + max_z))
    center_y = int(.5 * (min_y + max_y))
    center_x = int(.5 * (min_x + max_x))
    crop_center = np.array([center_z, center_y, center_x])

    for k, v in img_dict.items():
        if not isinstance(v, np.ndarray):
            # skip 'plan' entry basically
            continue

        img_dict[k] = crop_with_padding3d(v, crop_center, crop_size)
        print("Cropped", k, img_dict[k].shape)

    return img_dict


def select_ctv_contour(dose, contours, rel_error_thresh=0.05):
    """
    we compute the mean dose within each CTV contour and return
    the contour with the largest mean dose.

    Parameters
    ----------
    dose: 3d numpy array
        containing the dose volume of a plan
    contours: dict(str: np.array)
        containing CTV contour names and their binary
        segmentation masks (of same shape as dose)
    rel_error_meandose: float between 0 and 1
        The threshold on the relative error between the two
        contours with largest mean dose, below which we make
        a decision based on the volume instead of the mean
        dose
    """

    contour_names = list(contours.keys())
    if len(contour_names) == 0:
        raise ValueError("contours can not be empty!")
    elif len(contour_names) == 1:
        return contour_names[0]
    # from here on len(contours) >= 2

    # as we have multiple CTV contour candidates, we compute the mean
    # dose within each contour and take the contour with the largest
    # mean dose. If the two rois with largest mean doses are very similar
    # wrt mean dose, we take the one that covers more volume
    mean_doses = []
    for name in contour_names:
        mask = contours[name]
        mean_doses.append(
            dose[mask == 1].mean())

    # sort, highest mean dose come first
    sort_idx = np.argsort(mean_doses)[::-1]
    mean_doses_sorted = np.array(mean_doses)[sort_idx]
    contour_names_sorted = np.array(contour_names)[sort_idx]

    if mean_doses_sorted[0] <= 0:
        raise ValueError(
            "All mean doses are at most 0. Please check your data!")

    rel_error = (mean_doses_sorted[0] -
                 mean_doses_sorted[1]) / mean_doses_sorted[0]
    if rel_error < rel_error_thresh:
        # the two contours with largest mean dose are very similar
        # in terms of relative error, so we take the contour that
        # covers the larger volume instead
        vol_0 = contours[contour_names_sorted[0]].sum()
        vol_1 = contours[contour_names_sorted[1]].sum()
        if vol_0 >= vol_1:
            return contour_names_sorted[0]
        else:
            return contour_names_sorted[1]
    else:
        # the contours with the two largest mean doses differ in relative error
        # of mean dose by more than the threshold -> we pick the one with largest
        # mean dose
        return contour_names_sorted[0]


class LETFileDataset(Dataset):
    """
    A dataset class that provides directory names and paths
    to image data. Does not read the images!
    """

    def __init__(self,
                 data_dir,
                 plan_ids,
                 ct_filename="CT.npy",
                 dose_filename="plan_dose.npy",
                 let_filename="LETd.npy",
                 roi_filename="ROIs.npy",
                 plan_table_path=None
                 ):
        """
        Parameters
        ----------
        data_dir: str or Pathlib.Path
            pointing to a directory that contains one directory per plan
            which includes the required numpy files.
            NOTE that we assume that each directory within
            data_dir is a plan directory and no other directories should
            be contained!
        plan_ids: list of str
            a list of plan ids for which we read in the data. All other
            plans whose directories are available in 'data_dir' will be
            ignored.
        ct_filename: str or None
            Name of the file containing the CT data within
            each plans directory
        dose_filename: str
            Name of the file containing the Dose data within
            each plans directory
        let_filename: str
            Name of the file containing the LET data within
            each plans directory
        roi_filename: str
            Name of the file containing the ROI data within
            each plans directory
        plan_table_path: str, Path or None
            A path pointing to an excel file that contains additional
            information for each plan_id
            regarding CTV contour name, number of fractions etc...
        """

        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        if not data_dir.exists():
            raise ValueError(f"{data_dir} does not exist!")

        self.data_dir = data_dir

        self.plan_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and
             d.name in plan_ids])

        self.ct_filename = ct_filename
        self.dose_filename = dose_filename
        self.let_filename = let_filename
        self.roi_filename = roi_filename

        if plan_table_path is None:
            self.plan_table = None
        else:
            if not isinstance(plan_table_path, Path):
                plan_table_path = Path(plan_table_path)
            if not plan_table_path.exists():
                raise ValueError(f"{plan_table_path} does not exist!")
            # read all sheets of excel file and combine sheets to one df
            plan_table_dict = pd.read_excel(plan_table_path, sheet_name=None)
            plan_table = pd.concat([df for _, df in plan_table_dict.items()])

            # restrict plan_table to the given plan ids
            plan_table = plan_table[plan_table["plan_id"].isin(plan_ids)]
            self.plan_table = plan_table

    def __len__(self):
        return len(self.plan_dirs)

    def __getitem__(self, idx):
        """
        Returns
        -------
        a dictionary with paths to the CT, Dose, LET and ROI filenames
        """
        plan_dir = self.plan_dirs[idx]
        plan_id = plan_dir.name
        # make sure the files exist
        if self.ct_filename is not None:
            ct_file = plan_dir / self.ct_filename
        else:
            ct_file = None

        dose_file = plan_dir / self.dose_filename
        let_file = plan_dir / self.let_filename
        roi_file = plan_dir / self.roi_filename

        for file in [ct_file, dose_file, let_file, roi_file]:
            if file is None:
                continue
            if not file.exists():
                raise ValueError(f"{file} does not exist!")

        retval = {
            "plan_id": plan_id,
            "ct": ct_file,
            "dose": dose_file,
            "let": let_file,
            "roi": roi_file
        }

        if self.plan_table is not None:
            plan_details = self.plan_table[
                self.plan_table["plan_id"] == plan_id]

            retval["plan_details"] = plan_details

        return retval


class LETDatasetInMemory:
    """
    This is a dataset that attempts to read all the relevant numpy arrays into main memory
    on startup, in order to avoid slow file I/O during training. However, this might require
    a lot of memory.
    """

    def __init__(self,
                 data_dir,
                 plan_ids,
                 plan_table_path=None,
                 ct_filename="CT.npy",
                 dose_filename="plan_dose.npy",
                 let_filename="LETd.npy",
                 roi_filename="ROIs.npy",
                 # chosen to contain 100% of DD_PBS data bboxes
                 crop_size=(224, 320, 320),
                 return_rois=None,
                 preprocess_transform=get_preprocess_transforms(),
                 augmentation_transform=None,
                 clip_let_below_dose=None,
                 multiply_let_by_dose=False,
                 ct_window=(-500, 2000),
                 ct_normalisation=True,
                 drop_plans_with_missing_or_empty_rois=True
                 ):
        """
        Parameters
        ----------
        data_dir: str or Pathlib.Path
            pointing to a directory that contains one directory per plan
            which includes the required numpy files.
            NOTE that we assume that each directory within
            data_dir is a plan directory and no other directories should
            be contained!
        plan_ids: list of str
            a list of plan ids for which we read in the data. All other
            plans whose directories are available in 'data_dir' will be
            ignored.
        plan_table_path: str or Path or None
            A path pointing to an excel file that contains additional
            information for each plan_id
            regarding CTV contour name, number of fractions etc...
        ct_filename: str or None
            Name of the file containing the CT data within
            each plans directory. If None, no CT are going to be returned
        dose_filename: str
            Name of the file containing the Dose data within
            each plans directory
        let_filename: str
            Name of the file containing the LET data within
            each plans directory
        roi_filename: str
            Name of the file containing the ROI data within
            each plans directory
        crop_size: None or tuple of length 3
            Size of crop to be extracted.
            We first take the "External" contour, setting
            everything below the lowest z coordinate of the "BrainStem" contour
            to zero and compute the bounding box of the remaining contour.
            Crop center coordinates are determined by averaging lower and upper
            coordinates of the bounding boxes.
            If None, images will be returned as they are, potentially
            differing in size between plans.
        return rois: None or list of str
            A list of roi names that should be returned. Note that each plan
            is expected to have a roi within the numpy file.
            If None, no rois will be part of the returned dictionary of each item.
        preprocess_transform: None or monai.transforms.Compose instance.
            A pipeline of preprocessing steps that should be performed, such
            as e.g. clip CT values, convert to torch tensor etc...
            Will be executed after optionally cropping to brain region.
            If None, no further preprocessing will be performed.
        augmentation_transform: None or monai.transforms.Compose instance.
            A pipeline of data augmentations that should be performed.
            If None, no augmentations will be performe.
        clip_let_below_dose: None or float
            If a float, the LET will be clipped to zero in areas
            where the dose is lower than the provided threshold. This threshold is only applied after Dose normalization by mean dose in CTV, so this threshold needs to be set accordingly.
            If None, the LET will not be clipped.
        multiply_let_by_dose: bool
            Whether to multiple the LET by the dose to downweight
            areas of high LET that are uninteresting since they did
            not obtain any dose.
        ct_window: tuple of length two or None
            Lower and upper values for the CT window to use.
            CT values lower than ct_window[0] will be set to ct_window[0].
            CT values above ct_window[1] will be set to ct_window[1].
            If None, no windowing is performed.
        ct_normalisation: bool
            Whether to normalize the CT values to range [0, 1].
            This is performed after ct_windowing.
        drop_plans_with_missing_or_empty_rois: bool
            If this is True, in case at least a single ROI for a plan is either
            missing or empty, this plan will be skipped and wont
            be part of the dataset. Default is True.
        """

        self.file_dataset = LETFileDataset(
            data_dir=data_dir,
            plan_ids=plan_ids,
            ct_filename=ct_filename,
            dose_filename=dose_filename,
            let_filename=let_filename,
            roi_filename=roi_filename,
            plan_table_path=plan_table_path)

        self.crop_size = crop_size
        self.return_rois = return_rois
        self.preprocess_transform = preprocess_transform
        self.augmentation_transform = augmentation_transform
        self.clip_let_below_dose = clip_let_below_dose
        self.multiply_let_by_dose = multiply_let_by_dose

        if ct_window is not None:
            assert len(ct_window) == 2
        self.ct_window = ct_window
        self.ct_normalisation = ct_normalisation

        self.drop_plans_with_missing_or_empty_rois = drop_plans_with_missing_or_empty_rois
        # Now we use the file_dataset to get the paths to all files
        # and read that into a list of dictionaries
        self.data_dicts = self._read_data()
        self.plan_ids = sorted([d['plan_id'] for d in self.data_dicts])

    def _read_data(self):
        data_dicts = []
        plans_missing_rois = []  # due to missing ROIs
        pbar = tqdm(range(len(self.file_dataset)))
        for idx in pbar:
            sample_paths = self.file_dataset[idx]
            plan_id = sample_paths['plan_id']

            pbar.write(f"{plan_id}: Loading image data")
            # NOTE: they dont have a color channel, so shape is 3D
            dose = np.load(sample_paths['dose']).astype(np.float32)
            let = np.load(sample_paths['let']).astype(np.float32)

            rois = np.load(sample_paths['roi'], allow_pickle=True).item()
            if "CTV" not in rois:
                raise ValueError(f"{plan_id}: 'CTV' not in rois")
            # scale dose by dividing through the mean dose of CTV
            if "CTV_SiB" in rois:
                ctv_key = "CTV_SiB"
            else:
                ctv_key = "CTV"
            ctv = rois[ctv_key]  # boolean array
            pbar.write(f"{plan_id}: Use {ctv_key} for dose normalization.")
            mean_dose_ctv = dose[ctv].mean()
            dose /= mean_dose_ctv

            # CT might be unavailable if ct_filename is None
            if sample_paths["ct"] is not None:
                ct = np.load(sample_paths['ct']).astype(np.float32)
            else:
                ct = None

            if self.clip_let_below_dose is not None:
                pbar.write(
                    f"{plan_id}: Clip LET to zero where dose is below "
                    f"{self.clip_let_below_dose} Gy.")
                let[dose < self.clip_let_below_dose] = 0

            if self.multiply_let_by_dose:
                pbar.write(f"{plan_id}: Multiply LET by dose")
                let *= dose

            if ct is not None and self.ct_window is not None:
                low, high = self.ct_window
                pbar.write(f"{plan_id}: CT windowing to {low}, {high}")
                ct[ct < low] = low
                ct[ct > high] = high

            if ct is not None and self.ct_normalisation:
                pbar.write(f"{plan_id}: CT normalisation to unit range.")
                low = ct.min()
                high = ct.max()
                ct = (ct - low) / (high - low)

            data_dict = {
                "plan_id": plan_id,
                "dose": dose,
                "dose_was_divided_by": mean_dose_ctv,  # the normalization constant
                "let": let
            }
            if ct is not None:
                data_dict["ct"] = ct

            if "plan_details" in sample_paths:
                data_dict["plan_details"] = sample_paths["plan_details"]

            # optionally add some ROIs to the data_dict
            if self.return_rois is not None:
                pbar.write(f"{plan_id}: Extracting ROIs {self.return_rois}")

                missing_rois = {}
                for roi_name_to_extract in self.return_rois:
                    missing_or_empty = False

                    # check if ROI is missing
                    if roi_name_to_extract not in rois:
                        pbar.write(
                            f"[W]: {plan_id}: ROI {roi_name_to_extract} not found in"
                            f" {sorted(rois.keys())}.")
                        missing_or_empty = True
                    else:
                        # not missing, but could be empty
                        roi = rois[roi_name_to_extract]
                        if roi.sum() == 0:
                            pbar.write(
                                f"[W]: {plan_id}: {roi_name_to_extract} is available but empty!")
                            missing_or_empty = True

                        data_dict["Mask_" + roi_name_to_extract] = roi

                    missing_rois[roi_name_to_extract] = missing_or_empty

                if self.drop_plans_with_missing_or_empty_rois:
                    if any(missing_rois.values()):
                        plans_missing_rois.append(plan_id)
                        pbar.write(
                            f"{plan_id}: will be skipped due to missing ROIs!"
                        )
                        continue

            # optionally crop the data and add padding if needed (with minimum value)
            if self.crop_size is not None:
                # make sure we have 'External' and some kind of 'BrainStem'
                # contour available
                pbar.write(
                    f"{plan_id}: Cropping based on 'External' bounding box.")
                if "External" not in rois:
                    raise ValueError(f"{plan_id}: 'External' not in rois")
                external_roi = rois["External"]
                if "BrainStem" not in rois and "Brainstem" not in rois:
                    raise ValueError(
                        f"{plan_id}: Neither 'BrainStem' nor 'Brainstem' found in rois")
                elif "BrainStem" in rois:
                    brainstem_roi = rois['BrainStem']
                else:
                    # either only 'Brainstem' or both, 'Brainstem' and 'BrainStem'
                    brainstem_roi = rois['Brainstem']

                data_dict = create_crop_based_on_external_contour(
                    data_dict, external_roi, brainstem_roi, self.crop_size)

            # discard plans if there are empty rois
            roi_sizes = {
                k: np.sum(v) for k, v in data_dict.items() if k.startswith("Mask_")}
            has_empty_rois = not np.all([v > 0 for v in roi_sizes.values()])
            if self.return_rois is not None and has_empty_rois:
                empty_rois = [k for k, v in roi_sizes.items() if v == 0]
                pbar.write(f"[W]: {plan_id}: has empty ROIs (maybe due to cropping): "
                           f"{empty_rois}.")
                plans_missing_rois.append(plan_id)
                if self.drop_plans_with_missing_or_empty_rois:
                    pbar.write(f"[W]: {plan_id}: will be skipped due to empty ROIs!"
                               f"{empty_rois}.")
                    continue

            # apply preprocessing transforms from monai
            # to convert to torch tensor and add color channels
            # NOTE: the preprocess transforms only take care of the ct, let and dose.
            # I.e. if they get converted to torch tensors, the optionally read roi masks
            # will still be numpy arrays (but they are only used for inference anyway)
            if self.preprocess_transform is not None:
                pbar.write(f"{plan_id}: Apply preprocessing transformations")
                data_dict = self.preprocess_transform(data_dict)
                # NOTE: when monai transforms are applied, the tensors are not
                # torch.Tensor but monai.data.MetaTensor but they inherit from
                # torch.Tensor

            data_dicts.append(data_dict)
            pbar.write("\n")

        print(f"Successfully read data from {len(data_dicts)} plans.")

        if len(plans_missing_rois) > 0:
            print(
                f"The following {len(plans_missing_rois)} plans contained "
                f"missing/empty ROIs: {plans_missing_rois}. They might have "
                "been skipped if 'drop_plans_with_missing_or_empty_rois' = True")

        return data_dicts

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]

        # resizing and augmentation
        if self.augmentation_transform is not None:
            data_dict = self.augmentation_transform(data_dict)

        return data_dict


class NTCPDatasetInMemory:
    """
    This dataset is intended only for NTCP inference as this might require to read multiple Teilserien of a patient, make LET predictions for each
    and aggregate the obtained LETs for a final evaluation of
    side effect probability.
    It uses a LETDatasetInMemory that attempts to read all the relevant
    numpy arrays into main memory on startup, in order to avoid slow
    file I/O during inference. However, this might require a lot of memory. Besides, in contrast to the LETDataset, this class
    is not plan-centered but patient centered, i.e. for a given index, multiple dictionaries (corresponding to multiple plans) might be returned and is hence not compatible with the
    torch.utils.data.DataLoader!
    """

    def __init__(self,
                 data_dir,
                 plan_ids,
                 return_rois,
                 plan_table_path,
                 ct_filename="CT.npy",
                 dose_filename="plan_dose.npy",
                 let_filename="LETd.npy",
                 roi_filename="ROIs.npy",
                 # chosen to contain 100% of DD_PBS data bboxes
                 crop_size=(224, 320, 320),
                 preprocess_transform=get_preprocess_transforms(),
                 clip_let_below_dose=None,
                 multiply_let_by_dose=False,
                 ct_window=(-500, 2000),
                 ct_normalisation=True,
                 ):
        """
        Parameters
        ----------
        data_dir: str or Pathlib.Path
            pointing to a directory that contains one directory per plan
            which includes the required numpy files.
            NOTE that we assume that each directory within
            data_dir is a plan directory and no other directories should
            be contained!
        plan_ids: list of str
            a list of plan ids for which we read in the data. All other
            plans whose directories are available in 'data_dir' will be
            ignored.
        return rois: list of str
            A list of roi names that should be returned. Note that each plan
            is expected to have a roi within the numpy file.
            If None, no rois will be part of the returned dictionary of each item.
        plan_table_path: str or Path
            A path pointing to an excel file that contains additional
            information for each plan_id
            regarding CTV contour name, number of fractions etc...
        ct_filename: str or None
            Name of the file containing the CT data within
            each plans directory. If None, no CT are going to be returned
        dose_filename: str
            Name of the file containing the Dose data within
            each plans directory
        let_filename: str
            Name of the file containing the LET data within
            each plans directory
        roi_filename: str
            Name of the file containing the ROI data within
            each plans directory
        crop_size: None or tuple of length 3
            Size of crop to be extracted.
            We first take the "External" contour, setting
            everything below the lowest z coordinate of the "BrainStem" contour
            to zero and compute the bounding box of the remaining contour.
            Crop center coordinates are determined by averaging lower and upper
            coordinates of the bounding boxes.
            If None, images will be returned as they are, potentially
            differing in size between plans.
        preprocess_transform: None or monai.transforms.Compose instance.
            A pipeline of preprocessing steps that should be performed, such
            as e.g. clip CT values, convert to torch tensor etc...
            Will be executed after optionally cropping to brain region.
            If None, no further preprocessing will be performed.
        clip_let_below_dose: None or float
            If a float, the LET will be clipped to zero in areas
            where the dose is lower than the provided threshold.
            If None, the LET will not be clipped.
        multiply_let_by_dose: bool
            Whether to multiple the LET by the dose to downweight
            areas of high LET that are uninteresting since they did
            not obtain any dose.
        ct_window: tuple of length two or None
            Lower and upper values for the CT window to use.
            CT values lower than ct_window[0] will be set to ct_window[0].
            CT values above ct_window[1] will be set to ct_window[1].
            If None, no windowing is performed.
        ct_normalisation: bool
            Whether to normalize the CT values to range [0, 1].
            This is performed after ct_windowing.
        """

        self.plan_dataset = LETDatasetInMemory(
            data_dir=data_dir,
            plan_ids=plan_ids,
            plan_table_path=plan_table_path,
            ct_filename=ct_filename,
            dose_filename=dose_filename,
            let_filename=let_filename,
            roi_filename=roi_filename,
            crop_size=crop_size,
            return_rois=return_rois,
            preprocess_transform=preprocess_transform,
            augmentation_transform=None,
            clip_let_below_dose=clip_let_below_dose,
            multiply_let_by_dose=multiply_let_by_dose,
            ct_window=ct_window,
            ct_normalisation=ct_normalisation
        )

        # after reading in the data of all plans, we create a lookup
        # dictionary from patient ids to plans
        self.patient_id_to_plan_idx = {}
        for plan_idx in range(len(self.plan_dataset)):
            plan_dict = self.plan_dataset[plan_idx]
            plan_id = plan_dict["plan_id"]

            # NOTE: we assume that the first part before an '_' is the
            # patient id, and what is below will likely be the TS indicator
            patient_id = plan_id.split("_")[0]
            if patient_id not in self.patient_id_to_plan_idx:
                self.patient_id_to_plan_idx[patient_id] = [plan_idx]
            else:
                # NOTE: we don't store the plan id (str) but the index
                # of the plan within self.plan_dataset
                self.patient_id_to_plan_idx[patient_id].append(plan_idx)

        self.patient_ids = sorted(self.patient_id_to_plan_idx.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # retrieve all entries of that patient
        plan_idxs = self.patient_id_to_plan_idx[patient_id]

        plan_dicts = [self.plan_dataset[pidx] for pidx in plan_idxs]
        # check that all plan_dicts have the same keys
        # for all keys with torch tensors, we could concatenate entries

        return (patient_id, plan_dicts)
