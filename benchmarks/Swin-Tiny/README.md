# Swin-Tiny

## Install

Install PopRT.

## Download ONNX model

```
wget https://bytemlperf-modelzoo.tos-cn-beijing.volces.com/open_swim_transformer.tar
tar -xvf open_swim_transformer.tar
```

## Modify ONNX model bs from 4 to 24

```
python modify_bs.py
```

## Run

Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 24:
PopRT conversion: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00]
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:01<00:00]
Asyncronous avg Session Time : 3.528ms
Tput: 6801.8464343009055
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 0.0002148472412955016
mae: 0.0114006781950593
```
