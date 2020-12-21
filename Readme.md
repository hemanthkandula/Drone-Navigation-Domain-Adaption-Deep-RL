


### Domain Adaptation in Unmanned Aerial Vehicles Navigation and Obstacle Avoidance using Deep Reinforcement Learning


## Environment
 Downlaod indoor_updown environment. [Link to download](https://tufts.box.com/s/walg8vs7mjm6i3nnczlupd41vaymg8dy) Download env and place it under ```unreal_envs/indoor_updown```

## Requirements
```pip install requirements.txt```

## Training Source Task:

1. Edit in ```configs/config.cfg``` files
    - set Target: false
    - set data_collect: false
    - set mode : train

2. Run ``` python main.py ```


## Training Target Task:

1. Edit in ```configs/config.cfg``` files
    - set Target: true
    - set data_collect: false
    - set mode : train

2. Run ``` python main.py ```


## Collecting Source Task Dataset for adaption :

1. Edit in ```configs/config.cfg``` files
    - set Target: false
    - set data_collect: true
    - set mode : train

2. Run ``` python main.py ```

3. Stop once if data collection  is enough for training.
    32 images collected for every 100 training steps


## Collecting Target Task Dataset for adaption :

1. Edit in ```configs/config.cfg``` files
    - set Target: true
    - set data_collect: true
    - set mode : train

2. Run ``` python main.py ```

3. Stop once if data collection  is enough for training.
    32 images collected for every 100 training steps


## Domain Adaption:

1. Collect path for Source model encoder.ckpt,classifier.ckpt files.
2. data path is already set
3.  Run ``` python adda_main.py ```




## Retraining policy and value function Target Task:

1. Edit in ```configs/config.cfg``` files
    - set Target: true
    - set data_collect: false
    - set mode : train

2. Edit  ```configs/DeepQLearning.cfg``` file
    - set custom_load: true
    - set custom_load_path: 'adda/adapted_target/'


2. Run ``` python main.py ```



## Infer Target Task from scratch:

1. Edit in ```configs/config.cfg``` files
    - set Target: true
    - set data_collect: false
    - set mode : infer

2. Edit  ```configs/DeepQLearning.cfg``` file
    - set custom_load: true
    - set custom_load_path: scratch trained model here

2. Run ``` python main.py ```




## Infer adapted Target Task :

1. Edit in ```configs/config.cfg``` files
    - set Target: true
    - set data_collect: false
    - set mode : infer

2. Edit  ```configs/DeepQLearning.cfg``` file
    - set custom_load: true
    - set custom_load_path: path to retrained model after adaption

2. Run ``` python main.py ```
