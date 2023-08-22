# ViT

## Install

Install PopRT and requirements `pip install -r requirements.txt`.

## Run

1. Export ONNX model

```
python export_onnx.py

Check precision between Pytorch and ONNX:
mse: 1.3662788348606236e-12
mae: 8.902088666218333e-07
```

2. Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 8:
PopRT conversion: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:49<00:00]
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:53<00:00]
Asyncronous avg Session Time : 3.077ms
Tput: 2600.2590616358234
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 2.5533706775604514e-06
mae: 0.0012760016834363341
Run batch size 32:
PopRT conversion: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:40<00:00]
Graph compilation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:29<00:00]
Asyncronous avg Session Time : 10.293ms
Tput: 3109.043208996447
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 4.7500452637905255e-06
mae: 0.0016558620845898986
```
