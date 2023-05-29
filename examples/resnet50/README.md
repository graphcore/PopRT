# Benchmarks: Resnet50 inference with fp32/fp16/fp8/fp8_weight precision

This README describes how to run Resnet50 models in fp32/fp16/fp8/fp8_weight precision with custom operation for CV inference on IPU.

## Download Dataset

We put 4 images of the imagenet 2012 validation set in the `data` folder to ensure that the code can run correctly, and the labels of this dataset are sorted into `val_official_clean.csv`. If you want to test the accuracy of this model, please download the complete validation dataset [here](https://image-net.org/challenges/LSVRC/2012/index.php) and place them in `data` folder.

## Do Inference

This script will download the resnet50 model with torchvision, then convert it to onnx format with fp32/fp16/fp8/fp8_weight data type, and perform inference on the IPU.

To run Resnet50 inference with fp32 precision:

```
python test_resnet.py --model_dir model \
                      --model_run_type fp32
```

To run Resnet50 inference with fp16 precision:

```
python test_resnet.py --model_dir model \
                      --model_run_type fp16 \
                      --half_partial
```

To run Resnet50 inference with fp8 precision:

```
python test_resnet.py --model_dir model \
                      --model_run_type fp8 \
                      --half_partial
```

To run Resnet50 inference with fp8 weight:

```
python test_resnet.py --model_dir model \
                      --model_run_type fp8_weight \
                      --half_partial
```

You can also convert fp32 ResNet50 to FP8 and verify inference performance through the following script:

1. Ops occlusion:

```
poprt \
       --input_model model/resnet50.onnx \
       --output_model model/resnet50_fp8.onnx \
       --input_shape input=1,3,224,224 \
       --precision fp8 \
       --fp8_skip_op_names Conv_5,Conv_13,Gemm_126 \
       --fp8_params F143,F143,-3,-3 \
       --convert_version 11 \
       --export_popef \
       --ipu_version ipu21
python ../convert_compile_and_run.py --popef resnet50/executable.popef --batches_per_step 100
```

2. Auto pad (you can see the difference in popef format):

```
poprt \
       --input_model model/resnet50.onnx \
       --output_model model/resnet50_fp8_auto_pad.onnx \
       --input_shape input=1,3,224,224 \
       --precision fp8 \
       --compiler_options custom_patterns=PadConvChannel \
       --fp8_params F143,F143,-3,-3 \
       --convert_version 11 \
       --export_popef \
       --ipu_version ipu21
python ../convert_compile_and_run.py --popef resnet50/executable.popef --batches_per_step 100
```

| Model Size & Latency | pure-FP8 | Occ-FP8 | AutoPad-FP8 |
|----------------------|----------|---------|-------------|
| popef size           | 61 M     | 71 M    | 61 M        |
| run#1                | 0.384ms  | 0.400ms | 0.389ms     |
| run#2                | 0.364ms  | 0.397ms | 0.385ms     |
| run#3                | 0.346ms  | 0.375ms | 0.374ms     |
| run#4                | 0.355ms  | 0.389ms | 0.370ms     |
| run#5                | 0.353ms  | 0.403ms | 0.379ms     |

## Result

We keep the first `conv_7x7` as fp16, because the speed of fp8 `conv_7x7` is very slow, and set the last `Gemm` to fp16, it can improve some accuracy and hardly affect the speed. The acc and throughput are as follows.

If use real data and `overlapio` is enabled:

| Model Name\\Dataset\\bs                                  | bs1  | bs2  | bs4  | bs8   | bs16  | bs32  | bs64  | bs90  | bs96  | ILSVRC2012_val |
| ---------------------------------------------------------| ---- | ---- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | -------------- |
| ResNet 50                                                | 1759 | 2316 | 2864 | 3321  | 3756  | oom   | oom   | oom   | oom   | 76.14%         |
| FP16 ResNet 50                                           | 3529 | 5284 | 7030 | 9307  | 11393 | 13593 | 15249 | 16470 | oom   | 76.14%         |
| FP8 ResNet50(first conv and last gemm use fp16)          | 3997 | 6314 | 8749 | 12476 | 15554 | 18705 | 21525 | 23500 | 22295 | 75.89%         |
| FP8 ResNet50(first conv, conv_13 and last gemm use fp16) | 4011 | 6277 | 8737 | 12256 | 15480 | 18748 | 21532 | 23190 | oom   | 76.05%         |

If use dummy data and `IPU synthetic` is enabled:

| Model Name\\Dataset\\bs                                  | bs1  | bs2  | bs4  | bs8   | bs16  | bs32  | bs64  | bs90  | bs96  |
| -------------------------------------------------------- | ---- | ---- | ---- | ----- | ----- | ----- | ----- | ----- | ----- |
| ResNet 50                                                | 1796 | 2356 | 2957 | 3377  | 3903  | oom   | oom   | oom   | oom   |
| FP16 ResNet 50                                           | 3653 | 5466 | 7438 | 9805  | 11825 | 14240 | 16084 | 17105 | 16790 |
| FP8 ResNet 50                                            | 4125 | 5670 | 8589 | 11390 | 13814 | 16657 | 12989 | oom   | oom   |
| FP8 ResNet50(first conv use fp16)                        | 4113 | 6485 | 9291 | 13347 | 16384 | 19695 | 22778 | 24858 | 23579 |
| FP8 ResNet50(first conv and last gemm use fp16)          | 4126 | 6499 | 9334 | 13313 | 16334 | 19667 | 22886 | 24962 | 23695 |
| FP8 ResNet50(first conv, conv_13 and last gemm use fp16) | 4102 | 6496 | 9260 | 13364 | 16357 | 19730 | 22809 | 24498 | oom   |
