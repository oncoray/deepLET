import pytorch_lightning as pl
import torch

from monai.networks.nets import BasicUNet, FlexibleUNet, SegResNet, UNETR


class LETPredictorBase(pl.LightningModule):
    def __init__(self,
                 loss="mse",
                 use_ct=True,
                 learning_rate=1.e-4,
                 weight_decay=1.e-2,
                 weight_loss_by_dose=False,
                 **kwargs):
        """
        Parameters
        ----------
        loss: str
            One of 'mse' or 'mae' for mean squared error or
            mean absolute error, respectively.
            This is used as a loss function for training the
            model.
        use_ct: bool
            Whether to use the CT image as second input channel besides
            the dose to predict LET
        learning_rate: float
            Learning rate during network training.
        weight_decay: float
            Weight decay regularization for AdamW.
        weight_loss_by_dose: bool
            Whether we should multiply the voxelwise loss by the
            dose in order to emphasize correct LET predictions in
            high dose areas.
        kwargs: can be passed in derived classes in order to adjust
                the model architecture
        """
        super().__init__()
        self.save_hyperparameters()

        if loss == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
            self.metric_fn = torch.nn.L1Loss(reduction="mean")
            self.metric_name = "mae"
        elif loss == "mae":
            self.loss_fn = torch.nn.L1Loss(reduction="none")
            self.metric_fn = torch.nn.MSELoss(reduction="mean")
            self.metric_name = "mse"
        else:
            raise ValueError(f"Unknown loss {loss}. Choose 'mse' or 'mae'!")

        self.model = self._build_model()
        self.output_relu = torch.nn.ReLU()

    def _build_model(self):
        raise NotImplementedError

    def _create_model_input(self, data):
        if self.hparams.use_ct:
            input_img = torch.cat([data["dose"], data["ct"]], dim=1)
        else:
            input_img = data["dose"]

        return input_img

    def forward(self, batch):
        input_img = self._create_model_input(batch)
        output_raw = self.model(input_img)
        # LET is always >= 0, so we enforce this
        return self.output_relu(output_raw)

    def _shared_step_trainphase(self, batch, train_or_val):
        pred = self(batch)

        assert "let" in batch
        label = batch["let"]

        # unreduced loss, i.e. same shape as input, i.e. B, C, Z, Y, X
        loss = self.loss_fn(pred, label)
        # also metric computation
        metric = self.metric_fn(pred, label)

        # optional weighting
        if self.hparams.weight_loss_by_dose:
            assert "dose" in batch
            # TODO: should we normalize the dose for each sample
            # individually before applying them as weights?
            d = batch["dose"]
            # the dimensions to normalize over (all but batch)
            dims = list(range(1, len(d.shape)))

            norm_dose = (d - d.mean(dim=dims, keepdim=True)) / \
                d.std(dim=dims, keepdim=True)

            loss *= norm_dose

        # now manual reduction (not required for metric)
        loss = loss.mean()

        self.log(f"{train_or_val}_loss", loss,
                 logger=True,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True,
                 sync_dist=True,
                 batch_size=len(batch["plan_id"]))

        self.log(f"{train_or_val}_{self.metric_name}", metric,
                 logger=True,
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True,
                 sync_dist=True,
                 batch_size=len(batch["plan_id"]))

        if train_or_val == "val":
            self.log("hp_metric", loss,
                     logger=True,
                     on_epoch=True,
                     on_step=False,
                     prog_bar=False,
                     sync_dist=True,
                     batch_size=len(batch["plan_id"]))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=False)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        predictor_group = parent_parser.add_argument_group("LET Predictor")
        predictor_group.add_argument(
            "--loss",
            type=str,
            choices=["mse", "mae"],
            help="Loss function for network training."
        )
        predictor_group.add_argument(
            "--learning_rate",
            type=float,
            default=1.e-4,
            help="Learning rate for network training."
        )
        predictor_group.add_argument(
            "--weight_decay",
            type=float,
            default=1.e-2,
            help="Weight decay regularisation for the optimizer."
        )
        predictor_group.add_argument(
            "--weight_loss_by_dose",
            action="store_true",
            default=False,
            help="Weights the voxelwise loss by the delivered dose to obtain more accurate predictions "
                 "in high dose areas."
        )

        return parent_parser


class BasicUNetLETPredictor(LETPredictorBase):
    def __init__(self,
                 loss="mse",
                 use_ct=True,
                 learning_rate=1.e-4,
                 weight_decay=0.01,
                 weight_loss_by_dose=False,
                 # unet specific arguments
                 unet_feature_maps=(16, 16, 32, 64, 128, 16),
                 unet_activation=(
                     "LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 unet_normalisation=("instance", {"affine": True})
                 ):
        """
        Parameters
        ----------
        unet_feature_maps: list/tuple of int
            define the number of feature maps in each downsampling block of the unet.
            The number of downsampling layers is len(unet_feature_maps) - 2.
            - the first values corresponds to a Conv block consisting of two convolutions
              that does not downsample
            - the last value corresponds to the feature size after the last upsampling.
            - the remaining values define the feature maps of the downsampling blocks.
        unet_activation: str or tuple of (str, dict)
            Name of the nonlinear activation function, optionally together with
            a dictionary of hyperparameters for the function.
        unet_normalisation: str or tuple of (str, dict)
            Name and optionally options for the normalisation layers.
        """

        super().__init__(
            loss=loss,
            use_ct=use_ct,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss_by_dose=weight_loss_by_dose,
            # also pass the unet options since the base class
            # will store them in the self.hparams and we can
            # later access them to build the model
            unet_feature_maps=unet_feature_maps,
            unet_activation=unet_activation,
            unet_normalisation=unet_normalisation
        )

    def _build_model(self):
        return BasicUNet(
            spatial_dims=3,
            in_channels=2 if self.hparams.use_ct else 1,
            out_channels=1,
            features=self.hparams.unet_feature_maps,
            act=self.hparams.unet_activation,
            norm=self.hparams.unet_normalisation
        )

    @ staticmethod
    def add_model_specific_args(parent_parser):
        unet_group = parent_parser.add_argument_group("BasicUNet")
        unet_group.add_argument(
            "--unet_feature_maps",
            type=int,
            nargs=6,
            default=(16, 16, 32, 64, 128, 16),
            help="Channels in the UNet. First is for channel adjustment."
                 "Last is for after last upsampling. The four in between "
                 "are for the downsampling blocks."
        )

        return parent_parser


class FlexibleUNetLETPredictor(LETPredictorBase):
    """
    Much larger than BasicUNet because an efficientnet variant is
    used as the feature extraction backbone.
    """

    def __init__(self,
                 loss="mse",
                 use_ct=True,
                 learning_rate=1.e-4,
                 weight_decay=0.01,
                 weight_loss_by_dose=False,
                 # unet specific arguments
                 unet_decoder_channels=(128, 64, 32, 16, 8),
                 unet_backbone="efficientnet-b0",
                 unet_activation=(
                     "LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 unet_normalisation=("instance", {"affine": True})
                 ):
        """
        Parameters
        ----------
        unet_decoder_channels: list/tuple of int of length 5
            define the number of feature maps in each upsampling block of the unet.
        unet_backbone: str
            The variant of efficientnet to use. Choose from
            'efficientnet-b0',..., 'efficientnet-b8', 'efficientnet-l2'
        unet_activation: str or tuple of (str, dict)
            Name of the nonlinear activation function, optionally together with
            a dictionary of hyperparameters for the function.
        unet_normalisation: str or tuple of (str, dict)
            Name and optionally options for the normalisation layers.
        """

        super().__init__(
            loss=loss,
            use_ct=use_ct,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss_by_dose=weight_loss_by_dose,
            # also pass the unet options since the base class
            # will store them in the self.hparams and we can
            # later access them to build the model
            unet_decoder_channels=unet_decoder_channels,
            unet_backbone=unet_backbone,
            unet_activation=unet_activation,
            unet_normalisation=unet_normalisation
        )

    def _build_model(self):
        return FlexibleUNet(
            spatial_dims=3,
            in_channels=2 if self.hparams.use_ct else 1,
            out_channels=1,
            pretrained=False,
            decoder_channels=self.hparams.unet_decoder_channels,
            backbone=self.hparams.unet_backbone,
            act=self.hparams.unet_activation,
            norm=self.hparams.unet_normalisation,
            upsample="deconv"
        )

    @ staticmethod
    def add_model_specific_args(parent_parser):
        unet_group = parent_parser.add_argument_group("FlexibleUNet")
        unet_group.add_argument(
            "--unet_decoder_channels",
            type=int,
            nargs=5,
            default=(128, 64, 32, 16, 8),
            help="Channels in the upsampling part of UNet."
        )
        unet_group.add_argument(
            "--unet_backbone",
            type=str,
            choices=[
                'efficientnet-b0',
                'efficientnet-b1',
                'efficientnet-b2',
                'efficientnet-b3',
                'efficientnet-b4',
                'efficientnet-b5',
                'efficientnet-b6',
                'efficientnet-b7',
                'efficientnet-b8',
                'efficientnet-l2'
            ],
            default="efficientnet-b0",
            help="Name of the feature extraction backbone."
        )

        return parent_parser


class SegResNetLETPredictor(LETPredictorBase):
    """
    A resnet instead of a Unet.
    """

    def __init__(self,
                 loss="mse",
                 use_ct=True,
                 learning_rate=1.e-4,
                 weight_decay=0.01,
                 weight_loss_by_dose=False,
                 # resnet specific arguments
                 resnet_init_filters=16,
                 resnet_blocks_down=(1, 2, 2, 4),
                 resnet_activation=(
                     "LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 resnet_normalisation=("instance", {"affine": True})
                 ):
        """
        Parameters
        ----------
        resnet_init_filters: int
            Number of feature maps to start with.
        resnet_blocks_down: list/tuple of length 4
            Number of layers in each of the four blocks.
        resnet_activation: str or tuple of (str, dict)
            Name of the nonlinear activation function, optionally together with
            a dictionary of hyperparameters for the function.
        resnet_normalisation: str or tuple of (str, dict)
            Name and optionally options for the normalisation layers.
        """

        super().__init__(
            loss=loss,
            use_ct=use_ct,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss_by_dose=weight_loss_by_dose,
            # also pass the resnet options since the base class
            # will store them in the self.hparams and we can
            # later access them to build the model
            resnet_init_filters=resnet_init_filters,
            resnet_blocks_down=resnet_blocks_down,
            resnet_activation=resnet_activation,
            resnet_normalisation=resnet_normalisation
        )

    def _build_model(self):
        return SegResNet(
            spatial_dims=3,
            in_channels=2 if self.hparams.use_ct else 1,
            out_channels=1,
            init_filters=self.hparams.resnet_init_filters,
            blocks_down=self.hparams.resnet_blocks_down,
            blocks_up=(1, 1, 1),
            act=self.hparams.resnet_activation,
            norm=self.hparams.resnet_normalisation,
            upsample_mode="deconv"
        )

    @ staticmethod
    def add_model_specific_args(parent_parser):
        resnet_group = parent_parser.add_argument_group("SegResNet")
        resnet_group.add_argument(
            "--resnet_init_filters",
            type=int,
            default=16,
            help="Channels in first resnet block. Doubled in each following block."
        )
        resnet_group.add_argument(
            "--resnet_blocks_down",
            type=int,
            nargs=4,
            default=(1, 2, 2, 4),
            help="Number of layers in each of the four downsampling blocks of the Resnet."
        )

        return parent_parser


class UNETRLETPredictor(LETPredictorBase):
    """
    A vision transformer based UNet.
    """

    def __init__(self,
                 # unetr specific arguments
                 unetr_image_size,
                 unetr_feature_size=16,
                 unetr_hidden_size=192,
                 unetr_mlp_dim=768,
                 unetr_num_heads=12,
                 unetr_normalisation=("instance", {"affine": True}),
                 # base class arguments
                 loss="mse",
                 use_ct=True,
                 learning_rate=1.e-4,
                 weight_decay=0.01,
                 weight_loss_by_dose=False,
                 ):
        """
        Parameters
        ----------
        unetr_image_size: tuple/list of length 3
            Specify expected image input dimensions along
            z, y and x axis.
        unetr_feature_size: int
            Number of convolutional feature maps in the first
            encoding part of UNETR. Doubled in 3 following
            blocks.
        unetr_hidden_size: int
            Dimensionality that each image patch gets mapped to,
            i.e. 'token' dimension.
        unetr_mlp_dim: int
            Dimensionality of the MLP layers within each
            transformer block
        unetr_num_heads: int
            Number of attention heads in each transformer block.
        unetr_normalisation: str or tuple of (str, dict)
            Name and optionally options for the normalisation layers.
        """

        super().__init__(
            loss=loss,
            use_ct=use_ct,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss_by_dose=weight_loss_by_dose,
            # also pass the unetr options since the base class
            # will store them in the self.hparams and we can
            # later access them to build the model
            unetr_image_size=unetr_image_size,
            unetr_feature_size=unetr_feature_size,
            unetr_hidden_size=unetr_hidden_size,
            unetr_mlp_dim=unetr_mlp_dim,
            unetr_num_heads=unetr_num_heads,
            unetr_normalisation=unetr_normalisation
        )

    def _build_model(self):

        # assert that each dimension of image shape is
        # divisible by 16, required for patch creation.
        img_shape = self.hparams.unetr_image_size
        for i, elem in enumerate(img_shape):
            if elem % 16 != 0:
                raise ValueError(
                    f"Image dimension {i}({elem}) is not divisible by 16 "
                    "as required for UNETR patch creation.")

        return UNETR(
            spatial_dims=3,
            in_channels=2 if self.hparams.use_ct else 1,
            out_channels=1,
            img_size=self.hparams.unetr_image_size,
            feature_size=self.hparams.unetr_feature_size,
            hidden_size=self.hparams.unetr_hidden_size,
            mlp_dim=self.hparams.unetr_mlp_dim,
            num_heads=self.hparams.unetr_num_heads,
            pos_embed="conv",
            norm_name=self.hparams.unetr_normalisation,
            qkv_bias=False
        )

    @ staticmethod
    def add_model_specific_args(parent_parser):
        unetr_group = parent_parser.add_argument_group("UNETR")
        unetr_group.add_argument(
            "--unetr_feature_size",
            type=int,
            default=16,
            help="Number of feature maps to start from in Conv part of UNETR. Doubled in each block."
        )
        unetr_group.add_argument(
            "--unetr_hidden_size",
            type=int,
            default=192,
            help="Dimensionality of each patch embedding."
        )
        unetr_group.add_argument(
            "--unetr_mlp_dim",
            type=int,
            default=768,
            help="Target dimensionality of each patch embedding in MLP layers of Transformer blocks."
        )
        unetr_group.add_argument(
            "--unetr_num_heads",
            type=int,
            default=12,
            help="Number of attention heads."
        )

        return parent_parser


def collate_for_prediction(inputs):
    """
    A collate function for the dataloader used for making predictions.
    This ensures that only the tensors required of the model are put to
    the device, not the masks or the plan details.

    Parameters
    ----------
    inputs: list of length equal to batch_size

    Returns
    -------
    A dict with keys 'plan_id', 'dose', 'let' and 'ct' (if present) which are required
    to obtain predictions

    """
    # print(type(inputs))
    # print(type(inputs[0]))
    # print(inputs[0].keys())

    retval = {
        "plan_id": [inp['plan_id'] for inp in inputs],
        "dose": torch.stack([inp["dose"] for inp in inputs]),
        "let": torch.stack([inp["let"] for inp in inputs])
    }
    if "ct" in inputs[0]:
        assert all([("ct" in inp) for inp in inputs])
        retval["ct"] = torch.stack([inp["ct"] for inp in inputs])

    return retval


# TODO: equivariant UNet
