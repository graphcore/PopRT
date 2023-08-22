# BERT Base, BERT Base Pack, BERT Large, BERT Large Pack Benchmark

## Install

```
1. Install PopRT

2. pip install -r requirements.txt
```

## Run

1. BERT Base benchmark with a maximum sequence length of 128

```
python ./test_bert_base.py --batch_size 16
```

INFO:"{'accuracy/mae': 0.0062740673, 'accuracy/mse': 0.00019670575, 'batch_size': 16, 'compile_cost': 27.63799476623535, 'iteration': 1000, 'latency_max': 0.018744707107543945, 'latency_mean': 0.016533680983611174, 'latency_median': 0.016547203063964844, 'latency_min': 0.005711078643798828, 'log_type': 'BenchmarkOutput', 'onnx_file': 'model.onnx', 'precision': 'fp16', 'size': \[16, 128\], 'total_time_costs': 3.3214244842529297, 'tput': 4817.210228881298}"

```
python ./test_bert_base.py --batch_size 128
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.00678422, 'accuracy/mse': 0.00031418557, 'batch_size': 128, 'compile_cost': 85.83376049995422, 'iteration': 1000, 'latency_max': 0.1283864974975586, 'latency_mean': 0.11260646885937757, 'latency_median': 0.11274313926696777, 'latency_min': 0.03844094276428223, 'log_type': 'BenchmarkOutput', 'onnx_file': 'model.onnx', 'precision': 'fp16', 'size': \[128, 128\], 'total_time_costs': 22.568212270736694, 'tput': 5671.694260248187}"

2. BERT Base pack benchmark with a maximum sequence length of 128

```
python ./test_bert_base_pack.py --batch_size 16 --max_valid_num 40 --random_data
```

run the cmd above you will get the results below:

INFO:Pack bert_base, Batch Size: 16 Throughput: 8199.318532361893 samples/s, Latency : 0.12196135520935059 ms

3. BERT Large benchmark with a maximum sequence length of 128

```
python ./test_bert_large.py --batch_size 18 --bps 256
```

run the cmd above you will get the results below:

INFO:"{'accuracy/mae': 0.013268984, 'accuracy/mse': 0.00021964927, 'batch_size': 18, 'compile_cost': 56.09477758407593, 'iteration': 1000, 'latency_max': 0.06665754318237305, 'latency_mean': 0.060975178583964215, 'latency_median': 0.061066627502441406, 'latency_min': 0.018007993698120117, 'log_type': 'BenchmarkOutput', 'onnx_file': 'model.onnx', 'precision': 'fp16', 'size': \[18, 128\], 'total_time_costs': 12.221307039260864, 'tput': 1472.8375567502824}"

4. BERT Large benchmark with a maximum sequence length of 384

run the cmd above you will get the results below:

```
python ./test_bert_large_pack.py --batch_size 4  --max_valid_num 64 --random_data
```

INFO:Pack bert large Batch Size: 4 Throughput: 677.6602092192916 samples/s, Latency : 1.475665807723999 ms
