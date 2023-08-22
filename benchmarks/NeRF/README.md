# NeRF

## Install

Install PopRT and requirements `pip install -r requirements.txt`.

## Run

1. Export ONNX model

```
python export_onnx.py --config_yaml config.yaml

Check precision between TF and ONNX:
mse: 1.582564114477325e-11
mae: 2.30886698204813e-07
```

2. Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 1344:
PopRT conversion:  60%|█████████████████████████████████████████████████▊                                 | 3/5 [00:00<00:00]2023-07-06 09:23:01,954 WARNING poprt float_to_half.py:141] Clip StatefulPartitionedCall/ne_rf/concat_6:0 to range (-65500.0, 65500.0)
2023-07-06 09:23:01,957 WARNING poprt float_to_half.py:141] Clip StatefulPartitionedCall/ne_rf/BroadcastTo_1:0 to range (-65500.0, 65500.0)
PopRT conversion: 100%|███████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00]
Graph compilation: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [02:29<00:00]
Tput (800 * 800 / time_per_image): 303732.79715055705
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 0.002415041672065854
mae: 0.0052106184884905815
```
