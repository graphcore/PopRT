# Resnet50 inference benchmark

## Install

```
1. Install PopRT

2. pip install -r requirements.txt
```

## Run

1. To get the best throughput:

```
python ./test_resnet50.py --batch_size 80  --bps 128 --precision fp16  --enable_8bit_input --enable_overlapio
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.002250691, 'accuracy/mse': 8.225765e-06, 'batch_size': 80, 'compile_cost': 159.79228472709656, 'iteration': 1000, 'latency_max': 0.039835214614868164, 'latency_mean': 0.025618734779777948, 'latency_median': 0.02540874481201172, 'latency_min': 0.017881155014038086, 'log_type': 'BenchmarkOutput', 'onnx_file': 'resnet50.onnx', 'precision': 'fp16', 'size': \[3, 224, 224\], 'total_time_costs': 5.1475913524627686, 'tput': 15541.249202255634}"

2. To get the best latency

```
python ./test_resnet50.py --batch_size 1  --bps 1 --precision fp16 --eightbitio
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.002422316, 'accuracy/mse': 9.666424e-06, 'batch_size': 1, 'compile_cost': 45.619696378707886, 'iteration': 1000, 'latency_max': 0.011254072189331055, 'latency_mean': 0.0015032422673833502, 'latency_median': 0.001493215560913086, 'latency_min': 0.00044655799865722656, 'log_type': 'BenchmarkOutput', 'onnx_file': 'resnet50.onnx', 'precision': 'fp16', 'size': \[3, 224, 224\], 'total_time_costs': 0.3206348419189453, 'tput': 3118.812646857619}"
