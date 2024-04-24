#!/bin/bash --login
#SBATCH --job-name=let_segresnet
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem-per-cpu 25000
#SBATCH --gres=gpu:1
#SBATCH --time=38:00:00
#SBATCH --array 0-4

conda activate let

FOLD=$SLURM_ARRAY_TASK_ID

NUM_WORKERS=0
GPUS=1

# TODO: adjust
DATA_DIR="PATH_TO_YOUR_DATA"
# TODO: adjust to point to an xlsx file containing plan information
PLAN_TABLE_PATH="PATH_TO_YOUR_DATA_XLSX_FILE"

CT_FILENAME="CT.npy"
CT_WINDOW=(-500 2000)

DOSE_FILENAME="plan_dose.npy"
LET_FILENAME="LETd.npy"
ROI_FILENAME="ROIs.npy"
CROP_SIZE=(224 320 320)
CLIP_LET_BELOW_DOSE=0.04  # set to zero to not clip any let values as only those with dose < val will be clipped
LOSS="mae"
LEARNING_RATE=1.e-4
WEIGHT_DECAY=1.e-2
EPOCHS=200
BATCH_SIZE=1  # larger batch size caused CUDA Memory error
MODEL_TYPE="segresnet"

# only relevant if MODEL_TYPE="basic_unet"
UNET_FEATURE_MAPS=(16 16 32 64 128 16)
# only relevant if MODEL_TYPE="flex_unet"
UNET_DECODER_CHANNELS=(128 64 32 16 8)
UNET_BACKBONE="efficientnet-b0"
# only relevant if MODEL_TYPE="segresnet"
RESNET_INIT_FILTERS=16
RESNET_BLOCKS_DOWN=(1 2 2 4)
# only relevant if MODEL_TYPE="unetr"
UNETR_FEATURE_SIZE=8
UNETR_HIDDEN_SIZE=64
UNETR_MLP_DIM=256
UNETR_NUM_HEADS=8

DATA_SPLIT_DIR="../../../data/dd_pbs/data_splits"
TRAIN_ID_FILE=$DATA_SPLIT_DIR/5_fold_cv/train_plan_ids_fold_$FOLD.csv
VALID_ID_FILE=$DATA_SPLIT_DIR/5_fold_cv/val_plan_ids_fold_$FOLD.csv

OUTPUT_DIR_BASE=$HOME"/experiments/let_prediction_new/dd_pbs"
#OUTPUT_DIR=$OUTPUT_DIR_BASE/"cohort_with_mono_and_spr/CTspr-LETd"/"clip_let_below_"$CLIP_LET_BELOW_DOSE/$MODEL_TYPE/"fold_"$FOLD
OUTPUT_DIR=$OUTPUT_DIR_BASE/"Dose-LETd"/"clip_let_below_"$CLIP_LET_BELOW_DOSE/$MODEL_TYPE/"fold_"$FOLD
#OUTPUT_DIR=$OUTPUT_DIR_BASE/"cohort_with_mono_and_spr/Dose-LETd"/"no_let_clip_trainloss-weighted-by-dose"/$MODEL_TYPE/"fold_"$FOLD
TRAIN_OUTPUT_DIR=$OUTPUT_DIR/"training"

python train.py --default_root_dir $TRAIN_OUTPUT_DIR\
                --accelerator "gpu"\
                --devices $GPUS\
                --num_workers $NUM_WORKERS\
                --batch_size $BATCH_SIZE\
                --max_epochs $EPOCHS\
                --data_dir $DATA_DIR\
                --train_id_file $TRAIN_ID_FILE\
                --valid_id_file $VALID_ID_FILE\
                --ct_filename $CT_FILENAME\
                --dose_filename $DOSE_FILENAME\
                --let_filename $LET_FILENAME\
                --roi_filename $ROI_FILENAME\
                --crop_size ${CROP_SIZE[*]}\
                --clip_let_below_dose $CLIP_LET_BELOW_DOSE\
                --ct_window ${CT_WINDOW[*]}\
                --model_type $MODEL_TYPE\
                --loss $LOSS\
                --learning_rate $LEARNING_RATE\
                --weight_decay $WEIGHT_DECAY\
                --unet_feature_maps ${UNET_FEATURE_MAPS[*]}\
                --unet_decoder_channels ${UNET_DECODER_CHANNELS[*]}\
                --unet_backbone $UNET_BACKBONE\
                --resnet_init_filters $RESNET_INIT_FILTERS\
                --resnet_blocks_down ${RESNET_BLOCKS_DOWN[*]}\
                --unetr_feature_size $UNETR_FEATURE_SIZE\
                --unetr_hidden_size $UNETR_HIDDEN_SIZE\
                --unetr_mlp_dim $UNETR_MLP_DIM\
                --unetr_num_heads $UNETR_NUM_HEADS\
                --no_data_augmentation\
                # --use_ct\
                #--weight_loss_by_dose\
                #--multiply_let_by_dose\
                #--no_ct_normalisation\



INFERENCE_OUTPUT_DIR=$OUTPUT_DIR/"inference"
INFERENCE_ROIS=("Brain" "BrainStem" "CTV" "Chiasm" "OpticNerve_L" "OpticNerve_R" "Lens_L" "Lens_R" "LacrimalGland_L" "LacrimalGland_R")
# with head -n 4 | tail -1 we get the worst validation loss
# with head -n 2 | tail -1 we get (last, best val loss) and select best val loss
# with head -n 3 | tail -1 we get second best validation loss
# with head -n 1 | tail -1 we get 'last.ckpt'
CKPT_FILE=$(find $TRAIN_OUTPUT_DIR -type f -name \*.ckpt | sort -n | head -n 2 | tail -1)
VOXEL_AGGREGATION=("median" "mean" "max" "min" "1_percentile" "2_percentile" "98_percentile" "99_percentile")  # how voxel errors are aggregated to a measure per patient
# note: we can have unsigned / signed and relative / absolute
# unsigned_absolute: err = |gt - pred|
# signed_absolute:   err = (gt - pred)
VOXEL_ERROR_TYPE=("unsigned_absolute" "signed_absolute")
GPUS=1


INFERENCE_OUT_TRAIN=$INFERENCE_OUTPUT_DIR/"training"
python inference.py --output_dir $INFERENCE_OUT_TRAIN\
                    --data_dir $DATA_DIR\
                    --batch_size $BATCH_SIZE\
                    --gpus $GPUS\
                    --valid_id_file $TRAIN_ID_FILE\
                    --ct_filename $CT_FILENAME\
                    --dose_filename $DOSE_FILENAME\
                    --let_filename $LET_FILENAME\
                    --roi_filename $ROI_FILENAME\
                    --crop_size ${CROP_SIZE[*]}\
                    --clip_let_below_dose $CLIP_LET_BELOW_DOSE\
                    --ct_window ${CT_WINDOW[*]}\
                    --model_type $MODEL_TYPE\
                    --ckpt_file $CKPT_FILE\
                    --rois_to_evaluate ${INFERENCE_ROIS[*]}\
                    --voxel_aggregation ${VOXEL_AGGREGATION[*]}\
                    --voxel_error_type ${VOXEL_ERROR_TYPE[*]}\
                    # --use_ct\
                    # --multiply_let_by_dose\
                    # --no_ct_normalisation\


INFERENCE_OUT_VAL=$INFERENCE_OUTPUT_DIR/"validation"
python inference.py --output_dir $INFERENCE_OUT_VAL\
                    --data_dir $DATA_DIR\
                    --batch_size $BATCH_SIZE\
                    --gpus $GPUS\
                    --valid_id_file $VALID_ID_FILE\
                    --ct_filename $CT_FILENAME\
                    --dose_filename $DOSE_FILENAME\
                    --let_filename $LET_FILENAME\
                    --roi_filename $ROI_FILENAME\
                    --crop_size ${CROP_SIZE[*]}\
                    --clip_let_below_dose $CLIP_LET_BELOW_DOSE\
                    --ct_window ${CT_WINDOW[*]}\
                    --model_type $MODEL_TYPE\
                    --ckpt_file $CKPT_FILE\
                    --rois_to_evaluate ${INFERENCE_ROIS[*]}\
                    --voxel_aggregation ${VOXEL_AGGREGATION[*]}\
                    --voxel_error_type ${VOXEL_ERROR_TYPE[*]}\
                    # --use_ct\
                    # --multiply_let_by_dose\
                    # --no_ct_normalisation\


# NTCP models
NTCP_OUTPUT_DIR=$OUTPUT_DIR/"ntcp"

NTCP_OUT_TRAIN=$NTCP_OUTPUT_DIR/"training"
python ntcp.py --output_dir $NTCP_OUT_TRAIN\
               --data_dir $DATA_DIR\
               --batch_size $BATCH_SIZE\
               --gpus $GPUS\
               --valid_id_file $TRAIN_ID_FILE\
               --plan_table_path $PLAN_TABLE_PATH\
               --ct_filename $CT_FILENAME\
               --dose_filename $DOSE_FILENAME\
               --let_filename $LET_FILENAME\
               --roi_filename $ROI_FILENAME\
               --crop_size ${CROP_SIZE[*]}\
               --ct_window ${CT_WINDOW[*]}\
               --model_type $MODEL_TYPE\
               --ckpt_file $CKPT_FILE\
               # --clip_let_below_dose $CLIP_LET_BELOW_DOSE\  # this would clip the MC LET again
               # --use_ct\
               # --physical_dose\
               # --multiply_let_by_dose\
               # --no_ct_normalisation\


NTCP_OUT_VAL=$NTCP_OUTPUT_DIR/"validation"
python ntcp.py --output_dir $NTCP_OUT_VAL\
               --data_dir $DATA_DIR\
               --batch_size $BATCH_SIZE\
               --gpus $GPUS\
               --valid_id_file $VALID_ID_FILE\
               --plan_table_path $PLAN_TABLE_PATH\
               --ct_filename $CT_FILENAME\
               --dose_filename $DOSE_FILENAME\
               --let_filename $LET_FILENAME\
               --roi_filename $ROI_FILENAME\
               --crop_size ${CROP_SIZE[*]}\
               --ct_window ${CT_WINDOW[*]}\
               --model_type $MODEL_TYPE\
               --ckpt_file $CKPT_FILE\
               # --clip_let_below_dose $CLIP_LET_BELOW_DOSE\
               # --use_ct\
               # --physical_dose\
               # --multiply_let_by_dose\
               # --no_ct_normalisation\
