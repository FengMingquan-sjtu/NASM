# Optimal Control Operator Perspective and a Neural Adaptive Spectral Method (AAAI2025)

# Requirements
Install packages as
```
pip install -r requirements.txt
```
The Python version is 3.7.11

# Usage

Train our model on Pendulum using following command. 
```
python3 train.py --model_name NSMControl6  --system_name Pendulum
```
The output will be redirect to a log file named by date and time, e.g. `./log/2022-10-15 16:32:54.214730 `.  The dataset and benchmark set will be automatically generated and saved in `./data`, and the trained model will be periodically saved in `./model`. Use `python3 train.py -h` to check the list of supported models and systems.



# File/Folder Structure

1. `config.py` hyper-param configurations
2. `data_generator.py` generate data for supervised-learning setting
3. `evaluate.py` evaluate time and cost of trained models on benchmarks
4. `oc_model.py` torch implementation of deep models
5. `train.py` deep models training.
6. `utils` misc modules, e.g. logger, timer
7. `PDP/JinEnv.py` control environments definition
8. `PDP/PDP.py` direct method and PDP implementation
