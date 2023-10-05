### Notation
This is a simplifed version, only comsisting of toolbox and main code. Note that: the code will run over 10 hours, if we try all the folds.
### Environment 

Tensorflow 2.2 (CUDA=10.0) and Kapre 0.2.0. 

- option 1 (from yml)

```shell
conda env create -f V2S.yml
```

- option 2 (from clean python 3.6)

```shell
pip install tensorflow-gpu==2.1.0
pip install kapre==0.2.0
pip install h5py==2.10.0
pip install pyts
```

### How does the code work?


- LRCN_main.py:
To run it, please 
```shell
conda activate repr-adv
```
```shell
python LRCN_main.py --dataset 0 --eps 1000
```




