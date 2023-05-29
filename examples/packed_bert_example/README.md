# Packed Bert Base Squad

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

1). Prepare bert model

```bash
# get model

python -m transformers.onnx --model=csarron/bert-base-uncased-squad-v1 . --feature question-answering

# add position_ids for model

python add_position_ids.py --input_model model.onnx
```

2). Prepare original model (without packing)

```bash
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_base_bs16_sl256.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256
```

3). Prepare pack model

```bash
# add pack ops for pack model
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_base_bs16_sl256_pack.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256 --pack_args max_valid_num=40 segment_max_size=256 head_pattern='s:0->MatMul:1->Add:2->Split:3->Squeeze:4->e:5','Split:3->Squeeze:6->e:7' dynamic_groups=[[input_ids,]]

# pack mode with AttentionMask(input in pack will not across rows)
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_custom_attention_mask.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256 --passes insert_attention_mask
```

4). Execute the example

```bash
./packed_bert_example.py --model_with_packing_unpack_repack squad_bert_base_bs16_sl256_pack.onnx --model_without_packing squad_bert_base_bs16_sl256.onnx --model_with_packing_attention_mask squad_bert_custom_attention_mask.onnx
```

### Results

You will get performance of packed bert and original bert like:

```
[Original] Throughput: 1860.9792005501781 samples/s, Latency: 0.5373515188694 ms
....
[Pack Offline Unpack Repack] Throughput: 2542.8209428929054 samples/s, Latency: 0.3932640254497528 ms
....
[Pack Online Unpack Repack] Throughput: 2857.5999884178163 samples/s, Latency : 0.34994401037693024 ms
....
[Pack Offline AttentionMask] Throughput: 2782.587696947809 samples/s, Latency: 0.40753547847270966 ms
....
[Pack Online AttentionMask(first_fit)] Throughput: 3248.6212247330227 samples/s, Latency : 0.30782289803028107 ms
....
[Pack Online AttentionMask(next_fit)] Throughput: 2840.555334370472 samples/s, Latency : 0.35204383730888367 ms

```
