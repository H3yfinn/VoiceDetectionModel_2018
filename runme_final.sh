start=`date +%s`
WORKSPACE="/mnt/volume1/nov_data/8k_files/"
TEST_WAV_DIR=$WORKSPACE"testing"
TRAIN_WAV_DIR=$WORKSPACE"training"
TEST_CSV="metadata/testing.csv"
TRAIN_CSV="metadata/training.csv"
GT_WEAK_CSV="metadata/groundtruth_testing.csv"
GT_STR_CSV="metadata/groundtruth_testing.csv"
# You can to modify to your own workspace. 

# Extract features
python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir="/mnt/volume1/nov_data/features/logmel/testing" --recompute=True
python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir="/mnt/volume1/nov_data/features/logmel/training" --recompute=True  
python prepare_data.py pack_features --fe_dir="/mnt/volume1/nov_data/features/logmel/testing" --csv_path=$TEST_CSV --out_path="/mnt/volume1/nov_data/packed_features/logmel/testing.h5"
python prepare_data.py pack_features --fe_dir="/mnt/volume1/nov_data/features/logmel/training" --csv_path=$TRAIN_CSV --out_path="/mnt/volume1/nov_data/packed_features/logmel/training.h5"

# Calculate scaler
python prepare_data.py calculate_scaler --hdf5_path="/mnt/volume1/nov_data/packed_features/logmel/training.h5" --out_path="scalers/logmel/training.scaler"

# Train SED
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python main_crnn_sed.py train --tr_hdf5_path="/mnt/volume1/nov_data/packed_features/logmel/training.h5" --te_hdf5_path="packed_features/logmel/testing.h5" --scaler_path="scalers/logmel/training.scaler" --out_model_dir="models/crnn_sed_good"  --epochs=1 --model_name="19decAllFiles_1epoch"

# Recognize SED
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python main_crnn_sed.py recognize --te_hdf5_path="/mnt/volume1/nov_data/packed_features/logmel/testing.h5" --scaler_path="scalers/logmel/training.scaler" --model_dir="models/crnn_sed_good" --out_dir="preds/crnn_sed" --epochs=1

# Get stat of SED
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python main_crnn_sed.py get_stat --pred_dir="preds/crnn_sed" --stat_dir="stats/crnn_sed" --submission_dir="submissions/crnn_sed"  --gt_weak_csv=$GT_WEAK_CSV --gt_strong_csv=$GT_STR_CSV

end=`date +%s`

runtime=$((end-start))
echo "runtime: ", $runtime 
