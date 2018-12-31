from __future__ import print_function
import numpy as np
import sys
import soundfile
import os
import librosa
from scipy import signal
import pickle
import _pickle as cPickle
import scipy
import time
import csv
import gzip
import h5py
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import argparse
import time
import config as cfg

# Read wav
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
# Write wav
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

### Feature extraction. 
def extract_features(wav_dir, out_dir, recompute):
    """Extract log mel spectrogram features. 
    
    Args:
      wav_dir: string, directory of wavs. 
      out_dir: string, directory to write out features. 
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features. 
                 
    Returns:
      None
    """
    t_ex = time.time()
    print("FINN Extract started")
    
    fs = cfg.sample_rate
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))

    # Mel filter bank
    melW = librosa.filters.mel(sr=fs, 
                               n_fft=n_window, 
                               n_mels=64, 
                               fmin=0., 
                               fmax=4000.)#finn changed this from 8000 to 4000 
    #NB there is a chance that changing the numbers for this feature^ could be beneficial to performance
    
    cnt = 0
    for na in names:
        #print('finn, problem with Y being added before features names, trying to find solution', na)
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.p'
        
        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            if cnt % 1000 == 0:
                print('processing file number:', cnt, out_path)
            try:
                (audio, _) = read_audio(wav_path, fs)
            except:
                print("File %s is corrupted or not working" % wav_path)
                continue
            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute spectrogram
                ham_win = np.hamming(n_window)
                try:
                    [f, t, x] = signal.spectral.spectrogram(
                                x=audio, 
                                window=ham_win,
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                detrend=False, 
                                return_onesided=True, 
                                mode='magnitude') 
                except:
                    print("len(audio), audio.shape[-1], n_window, wav_path", len(audio), audio.shape[-1], n_window, wav_path)
                    continue
                x = x.T
                x = np.dot(x, melW.T)
                x = np.log(x + 1e-8)
                x = x.astype(np.float32)
                
                # Dump to pickle
                cPickle.dump(x, open(out_path, 'wb'), 
                             protocol=pickle.HIGHEST_PROTOCOL)#cpickle is often a cause of system issues. May have to change it to pickle on different versions
        cnt += 1
    print("FINN Extracting feature time: %s" % (time.time() - t_ex,), "cnt", cnt)

### Pack features of hdf5 file
def pack_features_to_hdf5(fe_dir, csv_path, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args: 
      fe_dir: string, directory of features. 
      csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the 
          string is empty, then pack features with all labels False. 
      out_path: string, path to write out the created hdf5 file. 
      
    Returns:
      None
    """
    
    t_pack= time.time()
    print("FINN Pack features started")
    
    
    max_len = cfg.max_len
    create_folder(os.path.dirname(out_path))
    
    #t1 = time.time()
    x_all, y_all, na_all = [], [], []
    
    if csv_path != "":    # Pack from csv file (training & testing from dev. data)
        with open(csv_path, 'rt') as f:
            reader = csv.reader(f)
            #print('this is where the error is', csv_path)
            lis = list(reader)
        cnt = 0
        for li in lis:
            [na, bgn, fin, lbs, ids] = li
            if cnt % 1000 == 0: print('packing feature number:', cnt)
            na = os.path.splitext(na)[0]
            fe_path2 = None
            fe_path3 = None
            #NB this stuff below is the cause of a lot of grr. Original process used weird formatting of names in metadata that I had to work around. #would make it moe simple but it seems it could just go with a full reset which i dont have time for
            if 'step_voices' in out_path:
                #make the process more simple to read i think.
                bgn1 = bgn.replace('.', '')
                fin1 = fin.replace('.', '') #you see, the training files dont have '.'s in their time values #fin = finish
                bgn1 = bgn1.replace(' ', '')#bgn = beginning
                fin1 = fin1.replace(' ', '')

                bare_na1 = na + '_' + bgn1 + '_' + fin1
                fe_na1 = bare_na1 + ".p"#dont know what 'fe' is an abbreviation for!
                fe_path1 = os.path.join(fe_dir, fe_na1)

                
                
                
                
            else:     
                


                bgn1 = bgn.replace('.', '')#bgn = beginning

                fin1 = fin.replace('.', '') #you see, the training files dont have '.'s in their time values #fin = finish
                bgn1 = bgn1.replace(' ', '')#bgn = beginning
                fin1 = fin1.replace(' ', '')
                bare_na1 = na + '_' + bgn1 + '_' + fin1 #old way of doing it. Dont think it works right
                fe_na1 = bare_na1 + ".p"#dont know what 'fe' is an abbreviation for!
                fe_path1 = os.path.join(fe_dir, fe_na1)

                bare_na2 = 'Y'+na+'_'+bgn+'_'+fin #ALSO! the testing files are supposed to have Y's at the start.. but they do ahve dots in their time values
                fe_na2 = bare_na2+".p"
                fe_path2=os.path.join(fe_dir, fe_na2)

                #can also have files with beginning = 0.000 and they are shown as 0
                bgn3 = '0'
                bare_na3 = na + '_' + bgn3 + '_' + fin1
                fe_na3 = bare_na3 + ".p"
                fe_path3 = os.path.join(fe_dir, fe_na3)
            
            if not os.path.isfile(fe_path2) and not os.path.isfile(fe_path1) and not os.path.isfile(fe_path3):#I'm pretty sure the only reason this can occur is if the feature wasnt extracted to begin with
                print("File %s is in the csv file but the feature is not extracted! ", bare_na1, ' ',  bare_na2)
            else:
                if os.path.isfile(fe_path2):
                    na_all.append(bare_na2+ ".wav") # Remove 'Y' in the begining. also finn-removed a '[1:]' from original code since it seemed unnecessary. Sincerely hoping this is unimportant
                    x = cPickle.load(open(fe_path2, 'rb'))#again, you may have cpickle issues
                elif os.path.isfile(fe_path1):
                    na_all.append(bare_na1+ ".wav")
                    x = cPickle.load(open(fe_path1, 'rb'))
                elif os.path.isfile(fe_path3):
                    na_all.append(bare_na3+ ".wav")
                    x = cPickle.load(open(fe_path3, 'rb'))
            #finn has made changes to all the above... beware! (:
                x = pad_trunc_seq(x, max_len)
                x_all.append(x)
                ids = ids.split(',')
                y = ids_to_multinomial(ids)
                y_all.append(y)
            cnt += 1
    else:   # Pack from features without ground truth label (dev. data) #not finns comment, not sure when this is enacted as well!
        print("FINN semi-ERROR: Packing from features without ground truth label (dev. data). Finn you wanted to know when this was happening")
        names = os.listdir(fe_dir)
        names = sorted(names)
        for fe_na in names:
            bare_na = os.path.splitext(fe_na)[0]
            fe_path = os.path.join(fe_dir, fe_na)
            na_all.append(bare_na + ".wav")
            try:
                x = cPickle.load(open(fe_path, 'rb'))
            except FileNotFoundError:
                print("FINN semi-ERROR: Packing from features without ground truth label (dev. data). Finn you wanted to know when this was happening")#this should now be looked at and a solution worked out
                #try:
                    #x = cPickle.load(open(fe_path2, 'rb'))
                #except FileNotFoundError:
                    #x = cPickle.load(open(fe_path3, 'rb'))
            #x = cPickle.load(open(fe_path, 'rb'))
            x = pad_trunc_seq(x, max_len)
            x_all.append(x)
            y_all.append(None)#hmm appending None to a list? seems fishy..
        
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.bool)
    print("len(na_all): %d", len(na_all))
    print("x_all.shape: %s, %s" % (x_all.shape, x_all.dtype))
    print("y_all.shape: %s, %s" % (y_all.shape, y_all.dtype))
    #changed na_all to type np.string because of type errors. #it was a really hard/annoying error! hopefully doesnt have a neg impact
    na_all = np.array(na_all, dtype=np.string_)#finn might need to include this to fix hp5y type errors
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('na_list', data=na_all)
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Save hdf5 to %s" % out_path)
    #print("Pack features time: %s" % (time.time() - t1,))
    print("FINN Packing feature time: %s" % (time.time() - t_pack,))
    
def ids_to_multinomial(ids):
    """Ids of wav to multinomial representation. 
    
    Args:
      ids: list of id, e.g. ['/m/0284vy3', '/m/02mfyn']
      
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    y = np.zeros(len(cfg.lbs))
    for id_ in ids:
        index = cfg.id_to_idx[id_]
        y[index] = 1
    return y
    
def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]
    return x_new
    
### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
        na_list = list(hf.get('na_list'))
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
        print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y, na_list

def calculate_scaler(hdf5_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    t1 = time.time()
    (x, y, na_list) = load_hdf5_data(hdf5_path, verbose=1)
    #print(hdf5_path, 'hdf5 path finn')
    #print(hdf5_path,type(x), len(x), x[0], x.shape())#finn
    (n_clips, n_time, n_freq) = x.shape
    x2d = x.reshape((n_clips * n_time, n_freq))
    scaler = preprocessing.StandardScaler().fit(x2d)
    print("Mean: %s" % (scaler.mean_,))
    print("Std: %s" % (scaler.scale_,))
    print("Calculating scaler time: %s" % (time.time() - t1,))
    pickle.dump(scaler, open(out_path, 'wb'))#todo finn are we going to have an issue with pickle here not being Cpickle? #dec4 it doesnt seem like any errors have occured.. weird!
    
def do_scale(x3d, scaler_path, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler_path: string, path of pre-calculated scaler. 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
    scaler = pickle.load(open(scaler_path, 'rb'))#finn i changed this from r to rb#dec4 i think this should potentially be looked into!#dec19 i changed it to r again
    #print("type(scaler), scaler) (perpare_data.py) delme dec1", type(scaler), scaler)
    #print(scaler)
    (n_clips, n_time, n_freq) = x3d.shape
    x2d = x3d.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x2d)
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))
    return x3d_scaled

### Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_ef = subparsers.add_parser('extract_features')
    parser_ef.add_argument('--wav_dir', type=str)
    parser_ef.add_argument('--out_dir', type=str)
    parser_ef.add_argument('--recompute', type=bool)
    
    parser_pf = subparsers.add_parser('pack_features')
    parser_pf.add_argument('--fe_dir', type=str)
    parser_pf.add_argument('--csv_path', type=str)
    parser_pf.add_argument('--out_path', type=str)
    
    parser_cs = subparsers.add_parser('calculate_scaler')
    parser_cs.add_argument('--hdf5_path', type=str)
    parser_cs.add_argument('--out_path', type=str)

    args = parser.parse_args()
    
    if args.mode == 'extract_features':
        extract_features(wav_dir=args.wav_dir, 
                         out_dir=args.out_dir, 
                         recompute=args.recompute)
    elif args.mode == 'pack_features':
        pack_features_to_hdf5(fe_dir=args.fe_dir, 
                              csv_path=args.csv_path, 
                              out_path=args.out_path)
    elif args.mode == 'calculate_scaler':
        calculate_scaler(hdf5_path=args.hdf5_path, 
                         out_path=args.out_path)
    else:
        raise Exception("Incorrect argument!")
