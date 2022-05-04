# klee_project_audio
This is an audio project for school (at Centralesupelec) with Klee group's partnership.

# Overview 
This project contains code for an audio gender counter. It also contains code to convert the final model into TF-Lite and ONNX. 

The hyper parameter search for the model could be found here : [Wandb](https://wandb.ai/mustapha/klee_project_audio_2/sweeps/82ldadx5?workspace=user-mustapha)

The model works fine in Raspberry Pi 3B+ 1GB with `on_raspberry.py` script in the `tf_model` folder.

The Arduino project is abandoned because of the final size of the model that couldn't fit in an Arduino nano 33 BLE's 256 KB available RAM. In fact, the TF-Lite Micro in arduino doesn't support LSTM opperations, so I had to unroll the model (90 KB to 300 KB).

# Usefull refs :
Please take a look at my [notion reference page](https://admitted-industry-353.notion.site/References-7f4e39f499a04d5bb919e7b8df767b2a)

# Data
### Linux or wsl
Just execute the bash script `get_data.sh` inside the data folder to get the data.

```bash
$ cd data
$ bash get_data.sh
```

# Content of the project
``` bash
    .
    ├── CountNet
    ├── LICENSE
    ├── README.md
    ├── assets
    ├── custom_models           # Submodule
    ├── data
    │   └── LibriCount
    ├── notebooks               # First notebooks, Pytorch version and HyperParameter search
    ├── old_model_conversion    # Converting old CountNet models into ONNX
    ├── requirements.txt
    ├── src                     # Dataloaders
    ├── tf_model                # Final TF model + ONNX and TF-Lite + Raspberry Pi
    │   ├── micro_speech        # Arduino project, abandoned !
    │   ├── onnx
    │   ├── saved 
    │   │   └── tf_model
    │   └── tmp
    └── wandb
```
## How to use
First get the model repository
```bash
$ git submodule init
$ git submodule update
```
Otherwise it is possible to pass this step if the flage `--recurse-submodules` was passed to the `git clone command`

#### Other
**The goal :**
Count the amount of concurrent speakers and detect their genders in an Audio.

**Dataset and model :**
The dataset and the model are both in the reference page : [notion reference page](https://admitted-industry-353.notion.site/References-7f4e39f499a04d5bb919e7b8df767b2a).
 
