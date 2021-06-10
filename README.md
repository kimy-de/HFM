
# Hidden Fluid Mechanics (Pytorch)


Hidden Fluid Mechanics (HFM) is a physics informed deep learning framework that extracts hidden valuable quantitative variables (e.g. velocity and pressure) using observations (e.g. concentration). 

[Reference] Maziar Raissi, Alireza Yazdani, and George Em Karniadakis. "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations." Science 367.6481 (2020): 1026-1030.

## 1. Training
```python
# Default hyperparameters
batch_size=10000
datapath='./data/Cylinder2D_flower.mat'
lr=0.0001
modelpath=None
num_samples=30000 # 157879
total_time=40
version_name='0'
```
```python
python train.py
```
```python
python train.py --modelpath './hfm_0.pth' # pretrained model
```

## 2. Evaluation
```python
python evaluation.py --modelpath './hfm_0.pth'
```
## 3. Results
Google Colab is used for the experiment. It tasks 25 hours to obtain the result.

Learning rate scheduler = [1e-3(~6 hours), 1e-4(~16 hours), 1e-5(~25 hours)]
<p align="center">
<img src="https://user-images.githubusercontent.com/52735725/121577522-e175dd80-ca29-11eb-93cd-d13bdb2cf0ed.gif" width="500">

</p>
