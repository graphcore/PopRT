#！/bin/bash

echo "Get bert model"
pip install -r ./requirements.txt
python -m transformers.onnx --model=csarron/bert-base-uncased-squad-v1 . --feature question-answering

echo "Add position_ids"
python add_position_ids.py --input_model model.onnx

echo "Convert origin model"
poprt --input_model model.onnx --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256 --precision fp16 --pack_args max_valid_num=32 segment_max_size=256

echo "Convert no pack model"
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_base_bs16_sl256.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256

echo "Convert pack model"
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_base_bs16_sl256_pack.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256 --pack_args max_valid_num=40 segment_max_size=256 head_pattern='s:0->MatMul:1->Add:2->Split:3->Squeeze:4->e:5','Split:3->Squeeze:6->e:7' dynamic_groups=[[input_ids,]]

echo "Convert pack model data not across row"
poprt --input_model bert_base_squad_pos.onnx --output_model squad_bert_custom_attention_mask.onnx --precision fp16 --input_shape input_ids=16,256 attention_mask=16,256 token_type_ids=16,256 position_ids=16,256 --passes insert_attention_mask

echo "Execute example"
python ./packed_bert_example.py --model_with_packing_unpack_repack squad_bert_base_bs16_sl256_pack.onnx --model_without_packing squad_bert_base_bs16_sl256.onnx --model_with_packing_attention_mask squad_bert_custom_attention_mask.onnx
