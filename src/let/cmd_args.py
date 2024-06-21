from argparse import ArgumentParser, ArgumentTypeError

def parse_list_of_tuples(input_str):
    """
    Parse a representation of a list of tuples.
    """

    try:
        parts = input_str.split(",")
        assert len(parts) == 4, "Parameter has to have four parts separated by ','"

        input_modality, loc_or_glob, dd, dta = parts
        assert input_modality in ["wedenberg", "bahn", "constant", "dose*LET", "LET"], f"First part '{input_modality}' not valid. Choose one of 'wedenberg', 'bahn', 'constant', 'LET' or 'dose*LET'"
        assert loc_or_glob in ["local", "global"], f"Second part '{loc_or_glob}' not valid. Choose 'local' or 'global'"

        dd = float(dd)
        dta = float(dta)

    except Exception as e:
        print(e)

        raise ArgumentTypeError("Parameter must be <rbe_method>,<local/global>,<dose_difference>,<distance_to_agreement>")

    return input_modality, loc_or_glob, dd, dta


def add_dataset_args(parser):
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="A directory that contains one directory per patient which "
             "includes the required numpy files."
    )
    group.add_argument(
        "--plan_table_path",
        type=str,
        help="A path to an excel file containing further info for each plan id (like number of fractions)."
    )
    group.add_argument(
        "--ct_filename",
        type=str,
        required=True,
        help="Name of the *.npy file containing the CT data for a patient."
    )
    group.add_argument(
        "--use_ct",
        action="store_true",
        default=False,
        help="Use CT as second input channel and concatenate with dose."
    )
    group.add_argument(
        "--dose_filename",
        type=str,
        required=True,
        help="Name of the *.npy file containing the dose data for a patient."
    )
    group.add_argument(
        "--let_filename",
        type=str,
        required=True,
        help="Name of the *.npy file containing the LET data for a patient."
    )
    group.add_argument(
        "--roi_filename",
        type=str,
        required=True,
        help="Name of the *.npy file containing the CT data for a patient."
    )
    group.add_argument(
        "--crop_size",
        type=int,
        nargs=3,
        default=None,
        help="Size of the image crop for each patient."
    )
    group.add_argument(
        "--clip_let_below_dose",
        type=float,
        default=None,
        help="Dose threshold in Gy that determines voxels below which the LET "
             "will be set to zero."
    )
    group.add_argument(
        "--multiply_let_by_dose",
        action="store_true",
        default=False,
        help="Multiply dose with LET to downweight unimportant voxels. "
             "Might simplify model training."
    )
    group.add_argument(
        "--ct_window",
        type=float,
        nargs=2,
        default=(-500., 2000.),
        help="Lower and upper bound for which to clip the CT data. "
             "Values below the first entry will be set to this value. "
             "Values above the second entry will be set to this value."
    )
    group.add_argument(
        "--no_ct_normalisation",
        action="store_true",
        default=False,
        help="CT values will not be normalised to unit range after windowing."
    )
    group.add_argument(
        "--physical_dose",
        action="store_true",
        default=False,
        help="Specify that doses that are read are physical doses."
             "Otherwise assume they are clinically RBE weighted doses "
             "with factor 1.1 multiplied to the physical dose."
    )

    return parser


def add_training_args(parser):
    group = parser.add_argument_group("Model training")
    group.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed (default: 1)")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1)
    group.add_argument(
        "--num_workers",
        type=int,
        default=0)
    group.add_argument(
        "--num_best_checkpoints",
        type=int,
        default=1,
        help="Number of best models to save as checkpoints."
    )
    group.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=None,
        help="Frequency to write out checkpoints."
    )
    group.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["basic_unet", "flex_unet", "segresnet", "unetr"]
    )
    group.add_argument(
        "--no_data_augmentation",
        action="store_true",
        default=False,
        help="Disable data augmentation during model training."
    )
    group.add_argument(
        "--train_id_file",
        type=str,
        required=True,
        help="Full path to a csv file containing the ids used for creating "
             "the training dataset.")
    group.add_argument(
        "--valid_id_file",
        type=str,
        default=None,
        help="Full path to a csv file containing the ids used for "
             "creating the validation dataset during training. "
             "If not set, no validation is done during model training.")
    group.add_argument(
        "--ckpt_file",
        type=str,
        required=False,
        help="Path to a checkpoint file of a pretrained model "
             "from which we can perform transfer learning.",
    )
    group.add_argument(
        "--retrain_only_last_layer",
        type=bool,
        default=False,
        help="Whether to train only the last layer with "
             "trainable parameters of a pretrained model. If this flag is not "
             "specified, all layers will be retrained. This flag has an effect "
             "only if a --ckpt_file is provided."
    )

    return parser


def add_inference_args(parser):
    group = parser.add_argument_group("Model inference")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    group.add_argument(
        "--ckpt_file",
        type=str,
        required=True,
        help="Path to the checkpoint file of the model to use.",
    )
    group.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["basic_unet", "flex_unet", "segresnet", "unetr"]
    )
    group.add_argument(
        "--rois_to_evaluate",
        type=str,
        nargs="+",
        required=True,
        help="Names of the ROIs for which we compute the prediction error."
    )
    group.add_argument(
        "--voxel_aggregation",
        type=str,
        nargs="*",
        choices=["median",
                 "mean",
                 "max",
                 "min",
                 "1_percentile",
                 "2_percentile",
                 "98_percentile",
                 "99_percentile"
                 ],
        default=["median"],
        help="How to aggregate voxel errors to patient errors."
    )
    group.add_argument(
        "--voxel_error_type",
        type=str,
        nargs="*",
        choices=["signed_absolute", "unsigned_absolute"],
        default=["signed_absolute"],
        help="Method to compute voxel errors. 'signed' means GT - PRED, "
             "while 'unsigned' means |GT - PRED|."
    )

    group.add_argument(
        "--execute_metric_computation",
        action="store_true",
        help="Flag to enable the calculation of classic metrics (DW,VW,percentiles,...). Argument should be set to perform computations."
    )

    group.add_argument(
        "--no_gamma",
        action="store_true",
        help="Flag to disable the gamma analysis. Argument should be set to not perform analysis."
    )

    parser.add_argument(
        '--gamma_configuration',
        type=parse_list_of_tuples,
        nargs="+",
        default="[]",
        help=(
            "A pythonic string respresentation of a list of tuples for gamma configuration."
            "Each tuple should have 4 elements: "
            "input_modality (str), gamma_type (str: 'local' or 'global'), "
            "dose_difference (float, in %), distance_to_agreement (float, in voxel size). "
            "Example: wedenberg,local,3.,3. wedenberg,local,2.,2."
            )
    )

    group.add_argument(
        "--use_gamma_multithreading",
        action="store_true",
        help="Flag to enable multithreading for gamma analysis. Argument should be set to use multithreading."
    )

    group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output path to store files."
    )
    group.add_argument(
        "--gpus",
        type=int,
        default=0
    )
    group.add_argument(
        "--valid_id_file",
        type=str,
        required=True,
        help="Full path to a csv file containing the ids used for "
             "inference."
    )

    group.add_argument(
        '--dose_rel',
        type=int,
        nargs='+',
        default=[0, 10, 25, 50, 75, 90, 100, 125],
        help='Relative dose bins relative to the mean CTV dose in %'
    )

    group.add_argument(
        '--let_bins_abs',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 15],
        help='Absolute LET bin edges'
    )
    # TODO: divide by 50 as we scale the dose by
    # mean dose in CTV
    group.add_argument(
        '--dose_bins_abs',
        type=float,
        nargs='+',
        # default=[0, 10, 20, 30, 40, 50, 60, 70],
        default=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        help='Absolute dose bin edges'
    )

    return parser


def add_ntcp_args(parser):
    group = parser.add_argument_group("NTCP model inference")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    group.add_argument(
        "--ckpt_file",
        type=str,
        required=True,
        help="Path to the checkpoint file of the model to use for LET prediction.",
    )
    group.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["basic_unet", "flex_unet", "segresnet", "unetr"]
    )
    group.add_argument(
        "--let_to_rbe_conversion",
        type=str,
        choices=["bahn", "wedenberg", "constant"],
        help="Formula to convert LET to RBE. "
             "Currently only 'bahn' is implemented."
    )
    group.add_argument(
        "--rbe_constant",
        type=float,
        default=1.1,
        help="When choosing let_to_rbe_conversion='constant'"
             " this defines the constant by which to multiply"
             " the dose to obtain an RBE weighted dose."
    )
    group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output path to store files."
    )
    group.add_argument(
        "--gpus",
        type=int,
        default=0
    )
    group.add_argument(
        "--valid_id_file",
        type=str,
        required=True,
        help="Full path to a csv file containing the ids used for "
             "inference."
    )

    return parser


def training_parser(title):
    parser = ArgumentParser(title)

    parser = add_dataset_args(parser)
    parser = add_training_args(parser)

    return parser


def inference_parser(title):
    parser = ArgumentParser(title)

    parser = add_dataset_args(parser)
    parser = add_inference_args(parser)

    return parser


def ntcp_parser(title):
    parser = ArgumentParser(title)

    parser = add_dataset_args(parser)
    parser = add_ntcp_args(parser)

    return parser
