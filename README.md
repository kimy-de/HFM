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
 <img width="800" alt="res1" src="https://user-images.githubusercontent.com/52735725/120918950-67ef8f80-c6b7-11eb-86dd-06576e075a3a.png">

</p>
