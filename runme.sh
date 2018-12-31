#!/bin/bash 
# You need to modify to your dataset path

start=`date +%s`

TEST_WAV_DIR="/mnt/volume1/25aug/5sept_testing_set"
TRAIN_WAV_DIR="/mnt/volume1/25aug/downloads25AUG"
EVALUATION_WAV_DIR="/vol/vssp/datasets/audio/audioset/task4_dcase2017_audio/official_downloads/evaluation"

# You can to modify to your own workspace. 
WORKSPACE=`pwd`

# Extract features
python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/testing" --recompute=True
python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/training" --recompute=True                                        
# Pack features
python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/testing" --csv_path="meta_data/5sept_testing_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/testing.h5"
python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/training" --csv_path="meta_data/5sept_training_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/training.h5"

# Calculate scaler
python prepare_data.py calculate_scaler --hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --out_path=$WORKSPACE"/scalers/logmel/training.scaler"

# Train SED
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main_crnn_sed.py train --tr_hdf5_path=$WORKSPACE"/packed_features/logmel/training.h5" --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --out_model_dir=$WORKSPACE"/models/crnn_sed"

# Recognize SED
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python main_crnn_sed.py recognize --te_hdf5_path=$WORKSPACE"/packed_features/logmel/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training.scaler" --model_dir=$WORKSPACE"/models/crnn_sed" --out_dir=$WORKSPACE"/preds/crnn_sed"

# Get stat of SED
end=`date +%s`

runtime=$((end-start))
echo runtime # does this work?
