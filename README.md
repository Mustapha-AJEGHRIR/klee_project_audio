# klee_project_audio
This is an audio project for school (at Centralesupelec) with Klee group's partnership



# Usefull refs :
Please take a look at my [notion reference page](https://admitted-industry-353.notion.site/References-7f4e39f499a04d5bb919e7b8df767b2a)

# Content of the project

    .
    ├── LICENSE
    ├── README.md
    └── countNet
        └── CountNet
            ├── Dockerfile
            ├── LICENSE
            ├── README.md
            ├── env.yml
            ├── eval.py
            ├── examples
            │   └── 5_speakers.wav
            ├── models
            │   ├── CNN.h5
            │   ├── CRNN.h5
            │   ├── F-CRNN.h5
            │   ├── RNN.h5
            │   └── scaler.npz
            ├── predict.py
            ├── predict_audio.py
            └── requirements.txt
## CountNet
#### Usage
First get the model repository
```bash
$ git submodule init
$ git submodule update
```
Get the dataset
```
```
#### Description
The goal :

    Count the amount of concurrent speakers in an Audio.

Dataset and model :

    The dataset and the model are both in the reference page.
