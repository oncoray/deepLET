import json
import pandas as pd
import pytorch_lightning as pl
import sys
import torch

from pathlib import Path
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint

from let.cmd_args import training_parser
from let.data import LETDatasetInMemory
from let.data_transform import get_preprocess_transforms,\
    get_augmentation_transforms
from let.model import LETPredictorBase, BasicUNetLETPredictor,\
    FlexibleUNetLETPredictor, SegResNetLETPredictor, UNETRLETPredictor


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    pl.seed_everything(args.seed, workers=True)

    training_ids = pd.read_csv(
        args.train_id_file,
        header=None).values.squeeze().tolist()

    if args.no_data_augmentation:
        train_aug = None
    else:
        train_aug = get_augmentation_transforms()

    train_dataset = LETDatasetInMemory(
        data_dir=args.data_dir,
        plan_ids=training_ids,
        ct_filename=args.ct_filename if args.use_ct else None,
        dose_filename=args.dose_filename,
        let_filename=args.let_filename,
        roi_filename=args.roi_filename,
        crop_size=args.crop_size,
        return_rois=None,  # not required during training
        preprocess_transform=get_preprocess_transforms(),
        augmentation_transform=train_aug,
        clip_let_below_dose=args.clip_let_below_dose,
        multiply_let_by_dose=args.multiply_let_by_dose,
        ct_window=args.ct_window,
        ct_normalisation=not args.no_ct_normalisation)
    print()
    print("len(train_dataset)", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    if args.valid_id_file is not None:
        valid_ids = pd.read_csv(
            args.valid_id_file,
            header=None).values.squeeze().tolist()

        val_dataset = LETDatasetInMemory(
            data_dir=args.data_dir,
            plan_ids=valid_ids,
            ct_filename=args.ct_filename if args.use_ct else None,
            dose_filename=args.dose_filename,
            let_filename=args.let_filename,
            roi_filename=args.roi_filename,
            crop_size=args.crop_size,
            return_rois=None,  # not required during training
            preprocess_transform=get_preprocess_transforms(),
            augmentation_transform=None,
            clip_let_below_dose=args.clip_let_below_dose,
            multiply_let_by_dose=args.multiply_let_by_dose,
            ct_window=args.ct_window,
            ct_normalisation=not args.no_ct_normalisation)

        print()
        print("len(val_dataset)", len(val_dataset))

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False)
    else:
        val_loader = None
        print("No validation data was provided.")

    if args.model_type == "basic_unet":
        model = BasicUNetLETPredictor(
            loss=args.loss,
            use_ct=args.use_ct,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            weight_loss_by_dose=args.weight_loss_by_dose,
            unet_feature_maps=args.unet_feature_maps,
        )
    elif args.model_type == "flex_unet":
        model = FlexibleUNetLETPredictor(
            loss=args.loss,
            use_ct=args.use_ct,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            weight_loss_by_dose=args.weight_loss_by_dose,
            unet_decoder_channels=args.unet_decoder_channels,
            unet_backbone=args.unet_backbone,
        )
    elif args.model_type == "segresnet":
        model = SegResNetLETPredictor(
            loss=args.loss,
            use_ct=args.use_ct,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            weight_loss_by_dose=args.weight_loss_by_dose,
            resnet_init_filters=args.resnet_init_filters,
            resnet_blocks_down=args.resnet_blocks_down,
        )
    elif args.model_type == "unetr":
        # we need to make sure that crop size is set
        # so that all patients have the same data shape.
        # otherwise a ViT can't work (in contrast to CNNs
        # which are shape agnostic if no FC layers are involved.)
        if args.crop_size is None:
            raise ValueError(
                "UNETR expects equal size of image input "
                "for each patient. Please specify 'crop_size' "
                "as a commandline argument.")
        model = UNETRLETPredictor(
            loss=args.loss,
            use_ct=args.use_ct,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            weight_loss_by_dose=args.weight_loss_by_dose,
            unetr_image_size=args.crop_size,
            unetr_feature_size=args.unetr_feature_size,
            unetr_hidden_size=args.unetr_hidden_size,
            unetr_mlp_dim=args.unetr_mlp_dim,
            unetr_num_heads=args.unetr_num_heads,
        )
    print(model)
    print(n_params(model) / 1.e6, "mio parameters")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{val_loss:.2f}-{train_loss:.2f}-{epoch:02d}",
        mode="min",
        save_last=True,
        save_top_k=args.num_best_checkpoints,
        every_n_epochs=args.checkpoint_every_n_epochs)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            checkpoint_callback
        ])

    # handle the case that a pretrained model weight file is specified
    if args.ckpt_file:
        print("Resume model training from given weights:"
              f" {args.ckpt_file}")
        pretrained_ckpt_file = args.ckpt_file

        # we only want to train the last layer
        if args.retrain_only_last_layer:
            trainable_children = list(
                c for c in model.model.children()
                if n_params(c) > 0)

            # deactivate requires_grad for all parameters
            model.freeze()
            model.train()  # undo the .eval() mode that happens in .freeze()

            last_trainable_layer = trainable_children[-1]
            print("Only training", last_trainable_layer)
            for param in last_trainable_layer.parameters():
                param.requires_grad = True

        print(n_params(model), "parameters will be fine-tuned")

    else:
        print("Training from scratch (random weights)!")
        pretrained_ckpt_file = None

    print()
    print("Start training")
    trainer.fit(
        model,
        train_loader, val_loader,
        ckpt_path=pretrained_ckpt_file)

    return 0


if __name__ == "__main__":
    parser = training_parser("LET model training")

    # register cmdline options for all models and lightning options
    model_classes = [
        LETPredictorBase,
        BasicUNetLETPredictor,
        FlexibleUNetLETPredictor,
        SegResNetLETPredictor,
        UNETRLETPredictor]
    for cls in model_classes:
        parser = cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print("Model training")
    print("\nParsed args are\n")
    pprint(args)

    # the output directory for the experiment
    default_root_dir = args.default_root_dir
    if default_root_dir is None:
        exp_name = args.model_type
        default_root_dir = f"./experiments/{exp_name}/training"
    if not isinstance(default_root_dir, Path):
        default_root_dir = Path(default_root_dir)
    if not default_root_dir.is_dir():
        default_root_dir.mkdir(parents=True)
    else:
        raise ValueError(
            f"Default_root_dir {default_root_dir} already exists!")

    print(f"\nUsing {default_root_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(default_root_dir / "commandline_args.json", 'w') as of:
        json.dump(vars(args), of, indent=2)

    retval = main(args)
    sys.exit(retval)
