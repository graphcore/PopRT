# Packed Deberta example

## PackRunner

PackRunner is a more efficient PopRT ModelRunner for dynamic sequences. PackRunner will pack input of different lengths together and feed it to IPU, it will pad zeros when reaches the maximum num of inputs (if needed) or run out of time.

## Quick Start Guide

### Prepare the environment

1). Prepare PopRT environment [PopRT User Guide](https://docs.graphcore.ai/projects/poprt-user-guide/en/latest/installation.html)

2). Install requirements

```bash
pip install -r requirements.txt
```

### Preprocess Model And Execute

1). Prepare deberta model

```bash
# get model
python -m transformers.onnx --model=IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese  . --feature sequence-classification
```

2). Prepare original model (without packing)

```bash
poprt --input_model model.onnx --output_model erlangshen_deberta_8_128.onnx --precision fp16 --input_shape input_ids=8,128 attention_mask=8,128 --disable_fast_norm --enable_insert_remap --remap_mode after_matmul,before_add --max_tensor_size 6291456
```

3). Prepare pack model

```bash
# pack mode with AttentionMask(input in pack will not across rows)
python ./modify_deberta.py erlangshen_deberta_8_128.onnx
```

4). Execute the example

```bash
python ./packed_deberta_example.py --model_without_packing erlangshen_deberta_8_128.onnx --model_with_packing_attention_mask packed_erlangshen_deberta_8_128.onnx
```

### Results

You will get performance of packed deberta and original deberta like:

```
[Original] Throughput: 1952.4748350216664 samples/s, Latency: 0.5121704936027527 ms

[Pack Offline AttentionMask] Throughput: 3089.4536026281285 samples/s, Latency: 0.3236818313598633 ms

[Pack Online AttentionMask:first_fit] Throughput: 3479.2992963515217 samples/s, Latency : 0.28741419315338135 ms

[Pack Online AttentionMask:next_fit] Throughput: 3080.640798822625 samples/s, Latency : 0.324607789516449 ms
```
