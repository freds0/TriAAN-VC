# -*- coding: utf-8 -*-
# Modifications
# 
# Original copyright:
# The copyright is under MIT license from VQMIVC.
# VQMIVC (https://github.com/Wendison/VQMIVC) / author: Wendison


import warnings
warnings.filterwarnings(action='ignore')

import os
from os.path import join as opj
from os import listdir

import json
import random
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import resampy
import pyworld as pw
import hashlib

from preprocess.spectrogram import logmelspectrogram

def ProcessingTrainData(path, cfg):
    
    """
        For multiprocess function binding load wav and log-mel 
    """
    wav_name = os.path.basename(path).split('.')[0]
    #speaker  = wav_name.split('_')[0]
    speaker = os.path.dirname(path).split('/')[-1]
    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db) # trim slience

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
        
    mel = logmelspectrogram(
                            x=wav,
                            fs=cfg.sampling_rate,
                            n_mels=cfg.n_mels,
                            n_fft=cfg.n_fft,
                            n_shift=cfg.n_shift,
                            win_length=cfg.win_length,
                            window=cfg.window,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax
                            )
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    return wav_name, mel, lf0, mel.shape[0], speaker

def LoadWav(path, cfg):
    
    """
        load raw wav from the path -> processed wav
    """
    # skip pre-emphasis
    wav_name = os.path.basename(path).split('.')[0]
    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db) # trim slience

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak

    return wav, wav_name

def GetLogMel(wav, cfg):

    """
        load log mel from the wav -> mel, f0, mel length
    """
    mel = logmelspectrogram(
                            x=wav,
                            fs=cfg.sampling_rate,
                            n_mels=cfg.n_mels,
                            n_fft=cfg.n_fft,
                            n_shift=cfg.n_shift,
                            win_length=cfg.win_length,
                            window=cfg.window,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax
                            )
    
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    return mel, lf0, mel.shape[0]


def NormalizeLogMel(wav_name, mel, mean, std):
    mel = (mel - mean) / (std + 1e-8)
    return wav_name, mel

def TextCheck(wavs, cfg):
    wav_files = [i.split('.wav')[0] for i in wavs]
    txt_path  = glob(f'{cfg.txt_path}/*/*')
    
    txt_files = [os.path.basename(i).split('.txt')[0] for i in txt_path]
    
    revised_wavs = []
    for i in range(len(wavs)):
        if wav_files[i] in txt_files:
            revised_wavs.append(wavs[i])
            
    return revised_wavs


def GetSpeakers(folder_path):
    all_spks = []
    for folder in listdir(folder_path):
            all_spks.append(folder)

    return all_spks


def SplitDataset(cfg):
    
    #all_spks = sorted(all_spks)
    #random.shuffle(all_spks)
    train_path = os.path.join(cfg.data_path, cfg.train_folder)
    train_spks = GetSpeakers(train_path)
    
    test_path = os.path.join(cfg.data_path, cfg.test_folder)
    test_val_spks = GetSpeakers(test_path)
    random.shuffle(test_val_spks)
    valid_spks, test_spks = test_val_spks[:len(test_val_spks)//2], test_val_spks[len(test_val_spks)//2:]
    
    print(f"Total {len(train_spks)} speakers in train set.")
    print(f"Total {len(valid_spks)} speakers in valid set.")
    print(f"Total {len(test_spks)} speakers in test set.")

    train_wavs_names = []
    valid_wavs_names = []
    test_wavs_names  = []
    
    for spk in train_spks:
        spk_wavs       = glob(os.path.join(train_path, f'{spk}/*'))
        spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        train_wavs_names += spk_wavs_names

    for spk in valid_spks:
        spk_wavs         = glob(os.path.join(test_path, f'{spk}/*'))
        spk_wavs_names   = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_wavs_names += spk_wavs_names

    for spk in test_spks:
        spk_wavs         = glob(os.path.join(test_path, f'{spk}/*'))
        spk_wavs_names  = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        test_wavs_names += spk_wavs_names
    
    all_wavs         = glob(f'{cfg.data_path}/*/*/*.wav')
    
    print(f'Total files: {len(all_wavs)}, Train: {len(train_wavs_names)}, Val: {len(valid_wavs_names)}, Test: {len(test_wavs_names)}, Del Files: {len(all_wavs)-len(train_wavs_names)-len(valid_wavs_names)-len(test_wavs_names)}')
    
    return all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names

def GetMetaResults(train_results, valid_results, test_results, cfg):
    """
    This is for making additional metadata [txt, text_path, test_type] -1:train, 0:s2s_st, 1:s2s_ut, 2:u2u_st, 3:u2u_ut
    """

    for i in range(len(train_results)):
        train_results[i]['test_type'] = 'train'
        
    train_spk = set([i['speaker'] for i in train_results])
    valid_spk = set([i['speaker'] for i in valid_results])
    test_spk  = set([i['speaker'] for i in test_results])

    valid_s2s_spk = train_spk.intersection(valid_spk) 
    valid_u2u_spk = valid_spk.difference(train_spk).difference(test_spk)

    test_s2s_spk  = train_spk.intersection(test_spk)
    test_u2u_spk  = test_spk.difference(train_spk).difference(valid_spk)

    for i in range(len(valid_results)):
        
        spk = valid_results[i]['speaker']
        if spk in valid_s2s_spk:
            valid_results[i]['test_type'] = 's2s_st'
        else:
            valid_results[i]['test_type'] = 'u2u_st'

    for i in range(len(test_results)):
        
        spk = test_results[i]['speaker']
        if spk in test_s2s_spk:
            test_results[i]['test_type'] = 's2s_st'

        else:
            test_results[i]['test_type'] = 'u2u_st'

    return train_results, valid_results, test_results


def ExtractMelstats(wn2info, train_wavs_names, cfg):
    
    mels = []
    for wav_name in train_wavs_names:
        mel, *_ = wn2info[wav_name]
        mels.append(mel)   
        
    mels      = np.concatenate(mels, 0)
    mean      = np.mean(mels, 0)
    std       = np.std(mels, 0)
    mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)    
    print('---Extract Mel statistics and save---')
    np.save(opj(cfg.output_path, 'mel_stats.npy'), mel_stats)
    
    return mean, std

def SaveFeatures(wav_name, info, mode, cfg):
    
    mel, lf0, mel_len, speaker = info
    folder_set = cfg.train_folder if mode == 'train' else cfg.test_folder
    wav_path      = os.path.join(cfg.data_path, folder_set, speaker, f'{wav_name}.wav') # can change to special char *
    mel_save_path = os.path.join(cfg.output_path, mode, 'mels', speaker, f'{wav_name}.npy')
    lf0_save_path = os.path.join(cfg.output_path, mode, 'lf0', speaker, f'{wav_name}.npy')                              
    
    os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(lf0_save_path), exist_ok=True)
    np.save(mel_save_path, mel)
    np.save(lf0_save_path, lf0)
    
    wav_name = wav_name.split('.wav')[0] # p231_001

    return {'mel_len':mel_len, 'speaker':speaker, 'wav_name':wav_name, 'wav_path':wav_path, 'mel_path':mel_save_path, 'lf0_path':lf0_save_path}
    
