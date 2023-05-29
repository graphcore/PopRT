# Benchmarks: BERT inference with fp32/fp16/fp8/fp8_weight precision

This README describes how to run BERT models in fp32/fp16/fp8/fp8_weight precision with custom operation for NLP inference on IPU.

## Download Dataset

These commands can be used to download the SQuAD dev set and evaluation script:

```
curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/dev-v1.1.json
curl -L https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py -o ./evaluate-v1.1.py
```

## Prepare the file of bert_config.json and vocab.txt

```
curl --create-dirs -L https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -o data/uncased_L-12_H-768_A-12.zip
```

Unzip the weights with:

```
unzip data/uncased_L-12_H-768_A-12.zip -d data
```

You will find the `bert_config.json` and `vocab.txt` in `data/uncased_L-12_H-768_A-12`

## Preprocessing the model

We need to do some preprocessing on `bertsquad-12.onnx` to ensure that this model can run correctly on IPU, you can download it with

```
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx
```

Then use the following command to convert the model:

```
poprt \
    --input_model bertsquad-12.onnx \
    --output_model bertsquad-12_fp32_bs_1.onnx  \
    --input_shape input_ids:0=1,256 input_mask:0=1,256 segment_ids:0=1,256 unique_ids_raw_output___9:0=1 \
    --convert_version 11
```

## Do Inference

This script will perform the inference of fp32/fp16/fp8 bert on the validation of SQuAD.

To run the inference of bert with SQuAD task in fp32 data type:

```
python test_bertsquad.py --model path_to_bertsquad-12_fp32_bs_1.onnx \
                         --model_save_path path_to_save_bert_fp32 \
                         --vocab_file path_to_vocab_file \
                         --bert_config_file path_to_bert_config \
                         --predict_file path_to_dev-v1.1.json \
                         --model_run_type fp32 \
                         --output_dir path_to_save_result
```

To run the inference of bert with SQuAD task in fp16 data type:

```
python test_bertsquad.py --model path_to_bertsquad-12_fp32_bs_1.onnx \
                         --model_save_path path_to_save_bert_fp16 \
                         --vocab_file path_to_vocab_file \
                         --bert_config_file path_to_bert_config \
                         --predict_file path_to_dev-v1.1.json \
                         --model_run_type fp16 \
                         --half_partial
                         --output_dir path_to_save_result
```

To run the inference of bert with SQuAD task in fp8 data type:

```
python test_bertsquad.py --model path_to_bertsquad-12_fp32_bs_1.onnx \
                         --model_save_path path_to_save_bert_fp8 \
                         --vocab_file path_to_vocab_file \
                         --bert_config_file path_to_bert_config \
                         --predict_file path_to_dev-v1.1.json \
                         --model_run_type fp8 \
                         --keep_precision_layer bert/embeddings/GatherV2 \
                         --output_dir path_to_save_result
```

To run the inference of bert with SQuAD task in fp8 weight data type:

```
python test_bertsquad.py --model path_to_bertsquad-12_fp32_bs_1.onnx \
                         --model_save_path path_to_save_bert_fp8 \
                         --vocab_file path_to_vocab_file \
                         --bert_config_file path_to_bert_config \
                         --predict_file path_to_dev-v1.1.json \
                         --model_run_type fp8_weight \
                         --keep_precision_layer bert/embeddings/GatherV2 \
                         --output_dir path_to_save_result
```

This script will get the accuracy.

```
evaluate-v1.1.py path_to_dev-v1.1.json \
                 path_to_save_result/predictions.json
```

## Result

The acc and throughput are as follow, this is the performance of Bert Base with 256 sequence length.

| Model Name          | exact_match | f1    | bs1 | bs2  | bs4  | bs8  | bs16 | bs32 | bs64 |
| ------------------- | ----------- | ----- | --- | ---- | ---- | ---- | ---- | ---- | ---- |
| bert_fp32           | 80.67       | 88.07 | 432 | 525  | oom  | oom  | oom  | oom  | oom  |
| bert_fp16           | 80.69       | 88.08 | 664 | 844  | 1068 | 1208 | 488  | oom  | oom  |
| bert_fp8            | 80.55       | 87.95 | 727 | 968  | 1239 | 1487 | 521  | 513  | oom  |
| bert_fp16_with_gelu | 80.66       | 88.05 | 766 | 1012 | 1352 | 1578 | 1657 | oom  | oom  |
| bert_fp8_with_gelu  | 80.44       | 87.94 | 857 | 1202 | 1660 | 2100 | 2122 | 1987 | 2540 |
