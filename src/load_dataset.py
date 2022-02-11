#%%
# -------------------------------- torch stuff ------------------------------- #
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------- other ---------------------------------- #
from glob import glob
from scipy.io import wavfile
import json
import os
from tqdm import tqdm
import sys

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



class AudioCountGender(Dataset):
    def __init__(self, data_dir=data_dir, dtype = FTYPE, cache=True, **kwargs):
        self.sounds = glob(os.path.join(data_dir,"*.wav"))
        self.labels = glob(os.path.join(data_dir,"*.json"))
        self.dtype = dtype
        self.cache = cache
        if self.cache:
            self.data = []
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
                self.data.append([torch.tensor(clip, dtype=self.dtype).unsqueeze(0), torch.tensor(genders, dtype=self.dtype)]) #unsqueeze serves for channel = 1
                
                
    def __getitem__(self, index):
        if self.cache:
            return self.data[index]
        else:
            # clip, sample_rate = sf.read(self.sounds[index])
            sample_rate, clip = wavfile.read(self.sounds[index])
            with open(self.labels[index]) as f:
                label = json.load(f)
            genders = [0, 0] #[Male, Female]
            for person in label:
                gender = person["sex"]
                if gender == "F":
                    genders[1] += 1
                else :
                    genders[0] += 1
            return torch.tensor(clip, dtype=self.dtype).unsqueeze(0), torch.tensor(genders, dtype=self.dtype) #unsqueeze serves for channel = 1
    def __len__(self):
        return len(self.sounds)


def get_splitter_dataloaders(**kwargs):
    """Return dataloaders for both training and validation sets,
    kmwargs could include:
        BATCH_SIZE
        TRAIN_SPLIT
        FTYPE

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
        
    data = AudioCountGender(**kwargs)
    train, val = torch.utils.data.random_split(data, [int(len(data)*train_split), len(data)-int(len(data)*train_split)])
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader    




