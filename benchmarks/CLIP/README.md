# CLIP

## Install

Install PopRT and requirements `pip install -r requirements.txt`.

## Run

1. Export ONNX model

```
python export_onnx.py

Check precision between Pytorch and ONNX:
mse: 7.093148984838038e-11
mae: 6.742477580701234e-06
```

2. Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 32:
PopRT conversion:  60%|█████████████████████████████████████████████████████████                                      | 3/5 [00:58<00:48]2023-07-04 04:20:45,531 WARNING poprt float_to_half.py:141] Clip 1651 to range (-65500.0, 65500.0)
PopRT conversion: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [01:18<00:00]
Graph compilation: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:51<00:00]
Asyncronous avg Session Time : 11.370ms
Tput: 2814.467977493102
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 0.000389773485949263
mae: 0.01577240414917469
Run batch size 56:
PopRT conversion:  60%|█████████████████████████████████████████████████████████                                      | 3/5 [00:42<00:35]2023-07-04 04:23:46,884 WARNING poprt float_to_half.py:141] Clip 1651 to range (-65500.0, 65500.0)
PopRT conversion: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:58<00:00]
Graph compilation: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:21<00:00]
Asyncronous avg Session Time : 13.737ms
Tput: 4076.631309287539
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 0.00033143628388643265
mae: 0.014250785112380981
```
