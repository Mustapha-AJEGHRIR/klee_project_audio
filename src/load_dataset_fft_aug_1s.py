#%%

"""
TODO
    - Add random shifts
    - Add random crop
"""


# -------------------------------- torch stuff ------------------------------- #
from curses import window
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

# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #

F16 = torch.float16
F32 = torch.float32
F64 = torch.float64


FTYPE = F32 # chosen, mainly because I hate getting nan values during training (when lr high and no grad clipping)
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
                dtype = FTYPE,
                fft_nperseg = 400,   # from the paper
                fft_noverlap = 240,  # from the paper
                fft_window_type = "tukey", # default
                fft_in_db = False,
                eps = 1e-8,
                shuffle = False, #made by the dataloader itself
                noise_attenuation = 0.1,
                add_noise = True,
                **kwargs):
        # ---------------------------------- config ---------------------------------- #
        self.dtype = dtype
        self.shuffle = shuffle
        self.noise_attenuation = noise_attenuation
        self.add_nosie = add_noise
        self.eps = eps
        self.fft_in_db = fft_in_db
        self.data = []
        self.sounds = sorted(glob(os.path.join(data_dir,"*.wav")))
        self.labels = sorted(glob(os.path.join(data_dir,"*.json")))
        if self.shuffle:
            self.sounds, self.labels = shuffe(self.sounds, self.labels)
        # ------------------------ load empty sounds from disk ----------------------- #
        self.empty_sounds = get_empty_audio(self.sounds)
        self.noise = []
        for sound in tqdm(self.empty_sounds, "Caching noise") :
            _, noise = wavfile.read(sound)
            _, _, fft_noise = scipy.signal.spectrogram(noise,
                                                        fs = sample_rate,
                                                        nperseg = fft_nperseg,
                                                        noverlap = fft_noverlap,
                                                        window = fft_window_type
                                                        )
            self.noise.append(torch.tensor(fft_noise, dtype=self.dtype).unsqueeze(0))
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
            self.data.append([torch.tensor(fft, dtype=self.dtype).unsqueeze(0), torch.tensor(genders)]) #unsqueeze serves for channel = 1
                
                
    def __getitem__(self, index):
        if self.add_noise:
            fft_noise = np.random.choice(self.noise) * self.noise_attenuation
        else :
            fft_noise = 0
        fft_mix = self.data[index][0] + fft_noise
        fft /= np.linalg.norm(fft_mix, axis=0, keepdims=True) + self.eps
        if self.fft_in_db:
            # fft = librosa.amplitude_to_db(fft, ref=np.max)
            fft = np.log(1 + fft) # the - is for not having negative values, the 50 is for some scaling (no very high values) 
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
        
    data = AudioCountGenderFft(**kwargs)
    # FIXME : I don't know if the torch utils loads first the data then split ? This is a problem for randomness
    train, val = torch.utils.data.random_split(data, [int(len(data)*train_split), len(data)-int(len(data)*train_split)])
    val.add_noise = validation_noise
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, data  




