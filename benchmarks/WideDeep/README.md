# WideDeep

## Install

Install PopRT and requirements `pip install -r requirements.txt`.

## Run

1. Export ONNX model

```
# Download TF saved model
wget -O open_wide_deep.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_wide_deep_saved_model.tar
tar xf open_wide_deep.tar

# Convert TF saved model to ONNX
python -m tf2onnx.convert --saved-model open_wide_deep_saved_model/ --output WideDeep.onnx

# Check the precision between TF and ONNX
python tf_vs_onnx.py
Check precision between TF and ONNX:
mse: 0.0
mae: 0.0
```

2. Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 8000:
PopRT conversion: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00]
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:25<00:00]
Asyncronous avg Session Time : 0.364ms
Tput: 21955275.518998653
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 8.741878165801244e-21
mae: 1.3060168181297516e-12
Run batch size 51200:
PopRT conversion: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00]
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:17<00:00]
Asyncronous avg Session Time : 1.440ms
Tput: 35545357.46289194
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 1.7104522762688272e-16
mae: 9.275135520736555e-11
```
