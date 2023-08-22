#!/bin/bash

echo "Install requirements"
pip install -r requirements.txt

echo "Export origin model"
python -m transformers.onnx --model=IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese  . --feature sequence-classification

echo "Convert model by poprt"
poprt --input_model model.onnx --output_model erlangshen_deberta_8_128.onnx --precision fp16 --input_shape input_ids=8,128 attention_mask=8,128 --disable_fast_norm --enable_insert_remap --remap_mode after_matmul,before_add --max_tensor_size 6291456

echo "Add AttentionMask and UnpackInfo ops"
python ./modify_deberta.py erlangshen_deberta_8_128.onnx

echo "Execute example"
python ./packed_deberta_example.py --model_without_packing erlangshen_deberta_8_128.onnx --model_with_packing_attention_mask packed_erlangshen_deberta_8_128.onnx
