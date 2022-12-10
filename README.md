# ASR with Deep Reinforcement Learning

Dedicated to the course project: Automatic Speech Recognition Using Deep Reinforcement Learning, for 11-785: Introduction to Deep Learning at Carnegie Mellon University, Fall 2022.

## CTC Decode Install

Run the following commands to install all packages in the given order

```
pip install -r requirements.txt

git clone --recursive https://github.com/parlance/ctcdecode.git
pip install wget
cd ctcdecode
pip install .
cd ..
```

## Instructions to run

To train baseline model:

```
python baseline_training.py
```

To fine tune policy gradient model:

```
python rl_training.py
```

## Files

1. baseline_modules.py - Definition of Baseline model
2. baseline_training.py - train and evaluate function and calls
3. dataloading.py - Datasets and Dataloader definitions
4. model_arch.py - Architecture of the baseline
5. multinomial_decoder.py - Multinomial Decoder definitions
6. rl_loss.py - Reinforcement Learning loss function
7. rl_training.py - train and evaluate function and calls
8. utils.py - levenshtein distance calculation code
