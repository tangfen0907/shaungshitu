# FuKAN
This repository contains the official implementation for the paper [Fuzzy KAN-Based Bidirectional Fast Anomaly Detection for Multi-Sensor Signals]().

## Requirements
The recommended requirements for FuKAN are specified as follows:
- arch==7.0.0
- einops==0.8.0
- hurst==0.0.5
- matplotlib==3.9.2
- numpy==1.26.4
- pandas==1.5.3
- scikit-learn==1.3.2
- scipy==1.13.1
- statsmodels==0.14.1
- torch==1.13.1
- tqdm==4.66.2
- tsfresh==0.20.3


The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data 
The datasets can be obtained and put into datasets/ folder in the following way:
- Our model supports anomaly detection for multivariate time series datasets.
- If you want to use your own dataset, please place your datasetfiles in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.
- For our datasets : - [SKAB](https://github.com/waico/SkAB) should be placed at `datasets/SKAB/`.
                              - [PUMP](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) should be placed at `datasets/PUMP/`.
                              - [WaDi](https://itrust.sutd.edu.sg/itrust-labs_datasets/) should be placed at `datasets/WaDi/`.
                              - [SWAT](https://drive.google.com/drive/folders/1ABZKdclka3e2NXBSxS9z2YF59p7g2Y5I) should be placed at `datasets/SWAT/`.


## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- main.py: The main python file. You can adjustment all parameters in there.
- metrics: There is the evaluation metrics code folder.
- model: FuKAN model folder
- solver.py: Another python file. The training, validation, and testing processing are all in there
- requirements.txt: Python packages needed to run this repo
## Usage
1. Install Python 3.6, PyTorch >= 1.4.0
2. Download the datasets
3. To train and evaluate FuKAN on a dataset, run the following command:
```bash
python main.py 
```
