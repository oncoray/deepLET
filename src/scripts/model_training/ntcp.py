import json
import numpy as np
import pandas as pd
import sys
import torch

from pathlib import Path
from pprint import pprint

from let.cmd_args import ntcp_parser
from let.data import NTCPDatasetInMemory, fuse_rois
from let.data_transform import get_preprocess_transforms
from let.model import BasicUNetLETPredictor,\
    FlexibleUNetLETPredictor, SegResNetLETPredictor, UNETRLETPredictor
from let import ntcp


def batch_and_send_to_device(plan_dict, device):
    """
    Adds a batch dimension for prediction and sends
    all tensor values required for prediction (exclude ROIs!)
    of the dictionary to the given device.
    """

    retval = {
        "plan_id": plan_dict["plan_id"]
    }

    required_keys = ["dose", "let"]
    if "ct" in plan_dict:
        required_keys.append("ct")

    for k in required_keys:
        assert k in plan_dict
        val = plan_dict[k]
        assert isinstance(val, torch.Tensor)
        if len(val.shape) != 5:
            val = val.unsqueeze(0)
            assert len(val.shape) == 5
        retval[k] = val.to(device)

    return retval


def get_aggregated_patient_data(model, dataset):
    """
    We would need a structure like this to loop effortlessly over the different LET->RBE conversions and NTCP models
    {
        'pat_1':
            'plan_doses': [...],
            'plan_let_mc': [...],
            'plan_let_dl': [...],
            'clinical_var_df': pd.DataFrame(a row for each plan, in order),
            'Mask_<ROI_NAME>': TODO: ROIs of which plan should we use??
    }
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    model = model.to(device)

    patient_data = {}
    for patient_idx, (patient_id, patient_plan_dicts) in enumerate(dataset):
        exclude_patient = False
        # sanity and conformity checks for a patient
        for plan_idx, plan_dict in enumerate(patient_plan_dicts):
            # we might need number of fractions etc from this
            if "plan_details" not in plan_dict:
                print(
                    f"[WW]: {patient_id}: {plan_dict['plan_id']}: dataset has no 'plan_details'! "
                    "Will exclude.")
                exclude_patient = True
                break

        if exclude_patient:
            print(f"{patient_id} will be excluded due to missing 'plan_details'!")
            continue

        # now check all plans if the ROI is really the same compared to first plan
        # with exception of CTV
        for plan_idx, plan_dict in enumerate(patient_plan_dicts):
            if plan_idx == 0:
                # we dont need to compare first plan with itself
                continue

            for k in plan_dict:
                if not k.startswith("Mask_"):
                    continue
                if "CTV" in k:
                    continue
                if not np.all(plan_dict[k] == patient_plan_dicts[0][k]):
                    print(
                        f"[WW]: {patient_id}, plan {plan_dict['plan_id']}: {k} "
                        f"differs from that of {patient_plan_dicts[0]['plan_id']}! "
                        "Will exclude!"
                    )
                    exclude_patient = True
                    break
            if exclude_patient:
                break

        if exclude_patient:
            print(
                f"{patient_id} will be excluded due to misaligned ROIs across plans!")
            continue

        n_plans = len(patient_plan_dicts)
        patient_data[patient_id] = {
            "plan_doses": [None] * n_plans,
            "plan_let_mc": [None] * n_plans,
            "plan_let_dl": [None] * n_plans,
            "clinical_var_df": [None] * n_plans
        }
        for plan_idx, plan_dict in enumerate(patient_plan_dicts):
            # we might need number of fractions etc from this

            # avoid putting the ROIs to device and only use plan_id, ct, dose and let
            plan_dict_no_rois = batch_and_send_to_device(plan_dict, device)
            let_dl = model(plan_dict_no_rois)[0].detach().cpu().numpy()

            # print(f"{patient_id}: plan_idx={plan_idx}, dose.shape={plan_dict['dose'].shape}")
            # get rid of the added batch dimension and color channels so everything is 3D np.arrays
            patient_data[patient_id]["plan_let_dl"][plan_idx] = let_dl[0]
            # we used normalized dose for obtaining model predictions, but
            # NTCP models require unnormalized doses!

            # the constant to multiply the 'dose' by to undo the dose scaling and get back
            # to the input dose before normalization
            dose_norm = plan_dict["dose_was_divided_by"]
            patient_data[patient_id]["plan_doses"][plan_idx] = plan_dict_no_rois["dose"].detach().cpu().numpy()[
                0, 0] * dose_norm
            patient_data[patient_id]["plan_let_mc"][plan_idx] = plan_dict_no_rois["let"].detach().cpu().numpy()[
                0, 0]
            patient_data[patient_id]["clinical_var_df"][plan_idx] = plan_dict["plan_details"]
        # add ROIs:
        # NOTE: for DD_PBS patients we checked that all ROIs corresponding
        # to patient anatomy are the same across plans, so we use it from first plan.
        # Only CTV is different across plans and we need to treat it differently.
        for k in patient_plan_dicts[0]:
            if not k.startswith("Mask_"):
                continue
            if "CTV" in k:
                continue
            patient_data[patient_id][k] = patient_plan_dicts[0][k]

        # handle CTVs of multiple plans
        # We fuse all of them. Reason is that in NTCP models likely
        # only the Brain - CTV, i.e. healthy brain is a ROI and we should
        # remove all voxels which are in at least one CTV (i.e. merge all CTV ROIs)
        all_ctv_rois = []
        print(f"{patient_id}: Fusing the following CTV rois:")
        for plan_idx, plan_dict in enumerate(patient_plan_dicts):
            for k in plan_dict:
                if "CTV" in k:
                    all_ctv_rois.append(plan_dict[k])
                    print(f"\tplan_idx={plan_idx}, contour_name={k}")
        patient_data[patient_id]["Mask_CTV"] = fuse_rois(all_ctv_rois)

    return patient_data


def evaluate_ntcp_models(model,
                         dataset,
                         ntcp_models,
                         dose_is_rbe_weighted,
                         alphabeta_per_roi,
                         rbe_constant=1.1):

    patient_data = get_aggregated_patient_data(model, dataset)

    results = []

    for descr, ntcp_cls in ntcp_models:
        ntcp_model = ntcp_cls()
        alphabeta = alphabeta_per_roi[ntcp_model.involved_roi]

        for let_to_rbe_conversion in ["bahn", "wedenberg", "constant"]:
            let_to_rbe_converter = ntcp.LETtoRBEConverter(
                let_to_rbe_conversion,
                alphabeta=alphabeta,
                rbe_constant=rbe_constant)

            for patient_id, patient_dict in patient_data.items():
                n_plans = len(patient_dict["plan_doses"])

                drbe_per_plan_mc = [None] * n_plans
                drbe_per_plan_dl = [None] * n_plans
                for pidx in range(n_plans):
                    drbe_mc = let_to_rbe_converter(
                        dose=patient_dict["plan_doses"][pidx],
                        let=patient_dict["plan_let_mc"][pidx],
                        dose_is_rbe_weighted=dose_is_rbe_weighted,
                        clinical_var_df=patient_dict["clinical_var_df"][pidx],
                    )
                    drbe_per_plan_mc[pidx] = drbe_mc

                    drbe_dl = let_to_rbe_converter(
                        dose=patient_dict["plan_doses"][pidx],
                        let=patient_dict["plan_let_dl"][pidx],
                        dose_is_rbe_weighted=dose_is_rbe_weighted,
                        clinical_var_df=patient_dict["clinical_var_df"][pidx])
                    drbe_per_plan_dl[pidx] = drbe_dl

                # sum up the rbe weighted doses of each plan to give us
                # a final RBE weighted dose of that patient, based on which
                # we can evaluate the NTCP models
                drbe_per_patient_mc = np.sum(drbe_per_plan_mc, axis=0)
                drbe_per_patient_dl = np.sum(drbe_per_plan_dl, axis=0)

                roi_dict = {k: v for k, v in patient_dict.items()
                            if k.startswith("Mask_")}

                ntcp_mc = ntcp_model(
                    dose=drbe_per_patient_mc,
                    roi_dict=roi_dict)

                ntcp_dl = ntcp_model(
                    dose=drbe_per_patient_dl,
                    roi_dict=roi_dict)

                results.append({
                    "patient": patient_id,
                    "NTCP_model": descr,
                    "alphabeta": alphabeta,
                    "let_to_rbe_conversion": let_to_rbe_conversion,
                    "NTCP_MonteCarlo": ntcp_mc,
                    "NTCP_Model": ntcp_dl,
                    "NTCP_error": ntcp_mc - ntcp_dl
                })

    return pd.DataFrame(results)


def main(args):
    # no seeding is necessary since inference does not involve randomness
    plan_ids = pd.read_csv(
        args.valid_id_file,
        header=None).values.squeeze().tolist()

    # NOTE: make sure to add all rois required for the ntcp models specified
    # below, otherwise they wont be read by the dataset
    rois_to_evaluate = [
        "Brain",                                # for MemoryG2At12m, MemoryG1At24m
        "BrainStem",                            # for FatigueG1At24m
        "CTV",                                  # for MemoryG2At12m, MemoryG1At24m
        "Chiasm",                               # for BlindnessAt60m
        "OpticNerve_L", "OpticNerve_R",         # for BlindnessAt60m
        "Lens_L", "Lens_R",                     # for CataractAt60m
        "LacrimalGland_L", "LacrimalGland_R",   # for OcularToxicityG2Acute
    ]

    # the third argument is for passing additional kwargs to the NTCP model during
    # prediction, such as e.g. the 'alphabeta' value if the Wedenberg model is used
    # (currently not implemented)
    # TODO: fix if wrong values come up here, I just filled them with the
    # same value we agreed on before.
    alphabeta_per_roi = {
        "Brain-CTV": 2.,
        "BrainStem": 2.,
        "Chiasm": 2.,
        "Cochlea": 2.,
        "Ipsilateral_LacrimalGland": 2.,  # Lisa in thesis used 3
        "Contralateral_LacrimalGland": 2.,
        "Lenses": 2.,  # Lisa in thesis used 1
        "Ipsilateral_OpticNerve": 2.,
        "Contralateral_OpticNerve": 2.,

    }

    ntcp_models = [
        ("Memory impairment G2@12m",
         ntcp.NTCPMemoryG2At12m,
         ),

        ("Memory impairment G1@24m",
         ntcp.NTCPMemoryG1At24m,
         ),

        ("Blindness (Chiasm) @60m",
         ntcp.NTCPBlindnessAt60mChiasm,
         ),

        ("Blindness (ON ipsi) @60m",
         ntcp.NTCPBlindnessAt60mONIpsi,
         ),

        ("Blindness (ON contra) @60m",
         ntcp.NTCPBlindnessAt60mONContra,
         ),

        ("Ocular toxicity ipsi G2@Acute",
         ntcp.NTCPOcularToxicityG2AcuteIpsi,
         ),

        ("Ocular toxicity contra G2@Acute",
         ntcp.NTCPOcularToxicityG2AcuteContra,
         ),

        # commented because most Essen patients do not have Lens_R roi
        ("Cataract ipsi @60m",
         ntcp.NTCPCataractAt60mIpsi,
         ),

        ("Cataract contra @60m",
         ntcp.NTCPCataractAt60mContra,
         ),

        ("Cataract merged @60m",
         ntcp.NTCPCataractAt60mMerged,
         )
    ]

    dataset = NTCPDatasetInMemory(
        data_dir=args.data_dir,
        plan_ids=plan_ids,
        return_rois=rois_to_evaluate,
        plan_table_path=args.plan_table_path,
        ct_filename=args.ct_filename if args.use_ct else None,
        dose_filename=args.dose_filename,
        let_filename=args.let_filename,
        roi_filename=args.roi_filename,
        crop_size=args.crop_size,
        preprocess_transform=get_preprocess_transforms(),
        clip_let_below_dose=args.clip_let_below_dose,
        multiply_let_by_dose=args.multiply_let_by_dose,
        ct_window=args.ct_window,
        ct_normalisation=not args.no_ct_normalisation)

    if args.model_type == "basic_unet":
        model_cls = BasicUNetLETPredictor
    elif args.model_type == "flex_unet":
        model_cls = FlexibleUNetLETPredictor
    elif args.model_type == "segresnet":
        model_cls = SegResNetLETPredictor
    elif args.model_type == "unetr":
        model_cls = UNETRLETPredictor
    else:
        raise ValueError(f"model_type {args.model_type} not understood!")

    model = model_cls.load_from_checkpoint(args.ckpt_file)
    model.eval()
    model.freeze()
    print(f"Loaded trained model from checkpoint {args.ckpt_file}.")

    metric_df = evaluate_ntcp_models(
        model,
        dataset,
        ntcp_models=ntcp_models,
        dose_is_rbe_weighted=not args.physical_dose,
        alphabeta_per_roi=alphabeta_per_roi)

    print(metric_df)
    for model, metric_df_for_model in metric_df.groupby("NTCP_model"):
        print(model)
        print(metric_df_for_model.describe())

    output_dir = Path(args.output_dir)
    metric_df.to_csv(
        output_dir / "ntcp_predictions.csv",
        index=False)

    return 0


if __name__ == "__main__":
    parser = ntcp_parser("LET prediction evaluation on NTCP models")

    args = parser.parse_args()
    print("NTCP Model evaluation")
    print("\nParsed args are\n")
    pprint(args)

    output_dir = args.output_dir
    if output_dir is None:
        exp_name = args.model_type
        output_dir = f"./experiments/{exp_name}/ntcp"
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    else:
        raise ValueError(
            f"output_dir {output_dir} already exists!")

    print(f"\nUsing {output_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(output_dir / "commandline_args.json", 'w') as of:
        json.dump(vars(args), of, indent=2)

    retval = main(args)
    sys.exit(retval)
