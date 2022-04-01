#%%

"""
TODO
    - Add random shifts
    - Add random crop
    - Add random speed modifications
    - Add random pitch modifications
"""


# -------------------------------- torch stuff ------------------------------- #
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------- other ---------------------------------- #
from glob import glob
from scipy.io import wavfile
import json
import os
from tqdm import tqdm
# import sys
# import librosa
import scipy
import scipy.signal
import numpy as np
# import copy
from sklearn.model_selection import train_test_split
# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #

F16 = torch.float16
F32 = torch.float32
F64 = torch.float64


FTYPE = F32 # chosen, mainly because I got some nan values during training (when lr high and no grad clipping)
TRAIN_SPLIT = 0.8
BATCH_SIZE = 64


data_dir = os.path.join(os.path.dirname(__file__),"../data/LibriCount")


def shuffe(sounds, labels):
    """Shuffle sounds and labels together
    """
    assert len(sounds) == len(labels)
    p = np.random.permutation(len(sounds))
    return sounds[p], labels[p]

def get_empty_audio(sounds):
    empty_sounds = []
    for sound in sounds:
        if "0_" in sound:
            empty_sounds.append(sound)
    return empty_sounds

class AudioCountGenderFft(Dataset):
    def __init__(self, data_dir=data_dir,
                split = "training",
                train_split = TRAIN_SPLIT,
                dtype = FTYPE,
                fft_nperseg = 400,   # from the paper
                fft_noverlap = 240,  # from the paper
                fft_window_type = "tukey", # default
                fft_in_db = False,
                eps = 1e-8,
                shuffle = False, #made by the dataloader itself
                noise_attenuation = 0.00001,
                add_noise = True,
                random_time_roll = True, # Will randomly shift the audio on the time axis (fft)
                max_random_frequency_roll = 1, # 1*40=40hz, Will randomly roll between -max_random_frequency_roll and max_random_frequency_roll
                **kwargs):
        # ---------------------------------- config ---------------------------------- #
        self.split = split
        self.random_time_roll = random_time_roll
        self.max_random_frequency_roll = max_random_frequency_roll
        self.dtype = dtype
        self.shuffle = shuffle
        self.noise_attenuation = noise_attenuation
        self.add_noise = add_noise
        self.eps = eps
        self.fft_in_db = fft_in_db
        self.data = []
        self.sounds = sorted(glob(os.path.join(data_dir,"*.wav")))
        self.labels = sorted(glob(os.path.join(data_dir,"*.json")))
        train_sounds, val_sounds, train_labels, val_labels = train_test_split(self.sounds, self.labels, train_size=train_split, random_state=42)
        if split == "training":
            self.sounds = train_sounds
            self.labels = train_labels
            
        elif split == "validation":
            self.sounds = val_sounds
            self.labels = val_labels
        else :
            raise ValueError("split must be either 'training' or 'validation'")
        
        if self.shuffle:
            self.sounds, self.labels = shuffe(self.sounds, self.labels)
        # ------------------------ load empty sounds from disk ----------------------- #
        self.empty_sounds = get_empty_audio(self.sounds)
        self.noise = []
        for sound in tqdm(self.empty_sounds, "Caching noise") :
            sample_rate, noise = wavfile.read(sound)
            _, _, fft_noise = scipy.signal.spectrogram(noise,
                                                        fs = sample_rate,
                                                        nperseg = fft_nperseg,
                                                        noverlap = fft_noverlap,
                                                        window = fft_window_type
                                                        )
            self.noise.append(torch.tensor(fft_noise, dtype=self.dtype))
        # ---------------------------- load data from disk --------------------------- #
        for index in tqdm(range(len(self.sounds)), "Caching dataset"):
            sample_rate, clip = wavfile.read(self.sounds[index])
            # assert self.sounds[index].split(".")[:-1] == self.labels[index].split(".")[:-1], "Sound and label files do not match in the order"            
            with open(self.labels[index]) as f:
                label = json.load(f)
            genders = [0, 0] #[Male, Female]
            for person in label:
                gender = person["sex"]
                if gender == "F":
                    genders[1] += 1
                else :
                    genders[0] += 1
                    
            _, _, fft = scipy.signal.spectrogram(clip,
                                                    fs = sample_rate,
                                                    nperseg = fft_nperseg,
                                                    noverlap = fft_noverlap,
                                                    window = fft_window_type
                                                    )
            self.data.append([torch.tensor(fft, dtype=self.dtype), torch.tensor(genders)]) #unsqueeze serves for channel = 1
                
                
    def __getitem__(self, index):
        if self.split == "training":
            if self.add_noise:
                random_i = np.random.randint(0, len(self.noise))
                fft_noise = self.noise[random_i] * self.noise_attenuation
            else :
                fft_noise = 0
            
            fft = self.data[index][0] + fft_noise

            if self.random_time_roll:
                max_roll = fft.shape[1]
                fft = np.roll(fft, np.random.randint(- max_roll, max_roll), axis=1)
            if self.max_random_frequency_roll > 0:
                fft = np.roll(fft, np.random.randint(- self.max_random_frequency_roll, self.max_random_frequency_roll), axis=0)
        else :
            fft = self.data[index][0]
        
        fft = fft / (np.linalg.norm(fft, axis=0, keepdims=True) + self.eps)
        if self.fft_in_db:
            # fft = librosa.amplitude_to_db(fft, ref=np.max)
            fft = np.log(1 + fft) # the - is for not having negative values, the 50 is for some scaling (no very high values) 
        # fft = torch.tensor(fft, dtype=self.dtype)
        return fft, self.data[index][1]
    
    def __len__(self):
        return len(self.sounds)


def get_splitter_dataloaders_fft(validation_noise = False, **kwargs):
    """Return dataloaders for both training and validation sets,
    kmwargs could include:
        BATCH_SIZE
        TRAIN_SPLIT
        FTYPE
        fft_nperseg
        fft_noverlap
        fft_window_type
        fft_in_db
        
    Returns:
        Dataloader: train_loader
        Dataloader: val_loader
    """
    
    if "BATCH_SIZE" in kwargs:
        batch_size = kwargs["BATCH_SIZE"]
    else :
        batch_size = BATCH_SIZE
    if "TRAIN_SPLIT" in kwargs:
        train_split = kwargs["TRAIN_SPLIT"]
    else:
        train_split = TRAIN_SPLIT
    
    # data = AudioCountGenderFft(**kwargs)
    # train, val = torch.utils.data.random_split(data, [int(len(data)*train_split), len(data)-int(len(data)*train_split)])
    train = AudioCountGenderFft(split = "training", train_split= train_split, **kwargs)
    val = AudioCountGenderFft(split = "validation", train_split= train_split, **kwargs)
    # # ---------------------- No perturbation for validation ---------------------- #
    # val.dataset.add_noise = validation_noise
    # val.dataset.random_time_roll = False
    # val.dataset.max_random_frequency_roll = 0
    # -------------------------------- Dataloaders ------------------------------- #
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, train, val




