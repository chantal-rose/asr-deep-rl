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
