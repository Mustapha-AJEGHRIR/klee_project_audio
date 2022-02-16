# klee_project_audio
This is an audio project for school (at Centralesupelec) with Klee group's partnership



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
```
    .
    ├── CountNet
    │   ├── Dockerfile
    │   ...
    │   └── requirements.txt
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── LibriCount
    │   ├── README.md
    │   └── get_data.sh
    └── notebooks
        └── EDA.ipynb
```
## CountNet
#### Usage
First get the model repository
```bash
$ git submodule init
$ git submodule update
```


#### Description
**The goal :**
Count the amount of concurrent speakers in an Audio.

**Dataset and model :**
The dataset and the model are both in the reference page.
