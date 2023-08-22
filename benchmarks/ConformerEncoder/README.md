# Conformer Enconder inference benchmark

## Install

```
Install PopRT

```

## Run

1. To get the best throughput:

```
python ./test_conformer_encoder.py --batch_size 32 --enable_overlapio
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.0018569251, 'accuracy/mse': 6.166293e-06, 'batch_size': 32, 'compile_cost': 112.47057151794434, 'iteration': 1000, 'latency_max': 0.03462028503417969, 'latency_mean': 0.019158532311608484, 'latency_median': 0.018984317779541016, 'latency_min': 0.012569904327392578, 'log_type': 'BenchmarkOutput', 'onnx_file': './/open_conformer/conformer_encoder.onnx', 'precision': 'fp16', 'size': \[3, 64, 512\], 'total_time_costs': 3.8601791858673096, 'tput': 8289.770619238807}"

2. To get the best latency

```
python ./test_conformer_encoder.py --batch_size 1 --enable_overlapio
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.0011900209, 'accuracy/mse': 2.4712324e-06, 'batch_size': 1, 'compile_cost': 43.4228515625, 'iteration': 1000, 'latency_max': 0.02205967903137207, 'latency_mean': 0.002796752316815717, 'latency_median': 0.002771139144897461, 'latency_min': 0.0018448829650878906, 'log_type': 'BenchmarkOutput', 'onnx_file': './/open_conformer/conformer_encoder.onnx', 'precision': 'fp16', 'size': \[3, 64, 512\], 'total_time_costs': 0.586573600769043, 'tput': 1704.8158981054096}"
