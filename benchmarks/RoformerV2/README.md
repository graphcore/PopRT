# RoformerV2

## Install

Install PopRT and requirements `pip install -r requirements.txt`.

## Run

1. Export ONNX model

```
python export_onnx.py

Check precision between Pytorch and ONNX:
mse: 5.465849994834571e-12
mae: 2.294778823852539e-06
```

2. Run PopRT

```
# Convert + Run
python poprt_run.py

Run batch size 1:
PopRT conversion:  60%|████████████████████████████████████████████████████████████▌                                        | 3/5 [00:05<00:04]2023-07-11 08:39:05,879 WARNING poprt float_to_half.py:123] Set Node /roformer/embeddings/LayerNorm/Constant_output_0 from 9.999999960041972e-13 to 1e-07
PopRT conversion: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:07<00:00]
Graph compilation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:13<00:00]
Asyncronous avg Session Time : 0.359ms
Tput: 2783.0665923507713
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 4.1110195070359623e-07
mae: 0.0005389750003814697
Run batch size 64:
PopRT conversion:  60%|████████████████████████████████████████████████████████████▌                                        | 3/5 [00:04<00:03]2023-07-11 08:39:27,970 WARNING poprt float_to_half.py:123] Set Node /roformer/embeddings/LayerNorm/Constant_output_0 from 9.999999960041972e-13 to 1e-07
PopRT conversion: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00]
Graph compilation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:29<00:00]
Asyncronous avg Session Time : 2.148ms
Tput: 29799.14388731994
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 6.331082659016829e-06
mae: 0.0019889669492840767
Run batch size 356:
PopRT conversion:  60%|████████████████████████████████████████████████████████████▌                                        | 3/5 [00:06<00:05]2023-07-11 08:40:10,913 WARNING poprt float_to_half.py:123] Set Node /roformer/embeddings/LayerNorm/Constant_output_0 from 9.999999960041972e-13 to 1e-07
PopRT conversion: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:08<00:00]
Graph compilation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:24<00:00]
Asyncronous avg Session Time : 8.348ms
Tput: 42643.069101505935
Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:
mse: 7.269463367265416e-06
mae: 0.00216607260517776
```
