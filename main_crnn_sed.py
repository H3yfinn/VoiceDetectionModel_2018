"""
Summary:  DCASE 2017 task 4 Large-scale weakly supervised 
          sound event detection for smart cars. Ranked 1 in DCASE 2017 Challenge.
Author:   Yong Xu, Qiuqiang Kong
Created:  03/04/2017
Modified: 31/10/2017
"""
from __future__ import print_function 
import sys
import _pickle as cPickle
import numpy as np
import argparse
import glob
import time
import os

from tensorflow.python import keras

from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random
import keras.optimizers
import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
from evaluation import io_task4, evaluate

# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

# Train model
def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))#removed this dec4 since its not helpful really

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    #print("delme dec 1, tr_x.shape", tr_x.shape)#output=51, 240, 64
    #print("delme dec 1, te_x.shape", te_x.shape)#:51, 240, 64
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])

    model = Model(input_logmel, out)
    model.summary()
    adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=adam_optimizer,
                  metrics=['accuracy'])#finn delme dec1 you can change this to categorical_accuracy to see if you can subvert the keras error. However you dont know if its the right hting to do. Keep a look out at the results to determineif it did what you wanted
    
    # Save model callback
    print("working here 1")
    filepath = os.path.join(args.out_model_dir, "{0}_{1}.hdf5".format(args.model_name, args.epochs)) 
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  
    
    # Data generator
        
    # Train
    t_train = time.time()
    print("FINN Training started")#this is really just me seeing if this is where most of the time is spent
    use_generator = False
    if use_generator:
        gen = RatioDataGenerator(batch_size=args.batch_size, type='train')#batch size should be manipulated from 44

        model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=args.steps_p_epoch,    # 100 iters is called an 'epoch'
                        epochs=args.epochs, #31             # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))
    else:
        model.fit(x=tr_x, y=tr_y, batch_size=20, epochs=args.epochs, verbose=1, callbacks=[save_model], validation_split=0.05, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=args.init_epoch, steps_per_epoch=None, validation_steps=None)
    model.save(os.path.join(args.out_model_dir, "final_model_{}_{}epochs.h5".format(args.model_name, args.epochs)))#am not sure if fit will save the final epoch.. pretty sure it does tho
    print("FINN Training finished, time taken: ", (time.time()-t_train))#this is really just me seeing if this is where most of the time is spent
    
# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in range(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

# Recognize and write probabilites. 
def recognize(args, at_bool, sed_bool):
    t_rec = time.time()
    print("FINN Recognize started")
    #print("recognize!")#todo remove
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    
    na_list = te_na_list
    #print("delme dec 1, num_classes", cfg.num_classes)#num_classes 3
    x = do_scale(x, args.scaler_path, verbose=1)

    fusion_at_list = []
    fusion_sed_list = []
    #for epoch in range(20, 30, 1):#hmm this value might need to be changed depending on nmber of epochs in model.fit_generator(... although this isnt a train session so maybe not!
    for epoch in range(1, args.epochs+1):#chane this when you want to increase epochs and reduce the amount of epochs pred'd over eg: range(20, epochs, 1)# range(1, args.epochs+1) is for args.epochs epochs btw
        #this allows you to go over all the epochs and average the scroe. but why?
        t1 = time.time()
        model_path = os.path.join(args.model_dir, "{0}_{1}.hdf5".format(args.model_name, epoch))
        print("model_path", model_path)
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            #dec4 this is where we find that shapes change from y having (#, n_classes) to (#, 2)... Dont know how to fix but i think its in changing the keras model
            #x.shape delme (22, 240, 64)
            #pred.shape delme (22, 2)
            t_pred = time.time()
            print("FINN at_pred started")
            pred = model.predict(x)
            print("FINN at_pred ended", (time.time()-t_pred))
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            t_pred = time.time()
            print("FINN pred_sed started")
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=20)
            fusion_sed_list.append(pred3d)
            print("FINN pred_sed ended", (time.time()-t_pred))
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("FINN Prediction finished!, time: ", (time.time()-t_rec))#this is really just me seeing if this is where most of the time is spent)

# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    print("Get Stat started!")
    t_stat = time.time()
    lbs = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    thres_ary = [0.001, 0.999]#[0.5] * len(lbs)

    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv=args.gt_weak_csv, #finn todo
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv=args.gt_strong_csv, #finn todo
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("FINN Calculating stat finished!, time: ", (time.time()-t_stat))#this is really just me seeing if this is where most of the time is spent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    parser_train.add_argument('--model_name', type=str)
    parser_train.add_argument('--epochs', type=int)
    parser_train.add_argument('--init_epoch', type=int)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--model_name', type=str)
    parser_recognize.add_argument('--out_dir', type=str)
    parser_recognize.add_argument('--epochs', type=int)
    
    parser_get_stat = subparsers.add_parser('get_stat')
    parser_get_stat.add_argument('--gt_weak_csv', type=str)
    parser_get_stat.add_argument('--gt_strong_csv', type=str)
    parser_get_stat.add_argument('--pred_dir', type=str)
    parser_get_stat.add_argument('--stat_dir', type=str)
    parser_get_stat.add_argument('--submission_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True, sed_bool=True)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=True)
    else:
        raise Exception("Incorrect argument!")
