# E2E example

This README describes how to run an End-to-End example with [GoogLeNet](https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/googlenet) based on PopRT.

## Env

```
pip install imageio
```

## Model

Download GoogLeNet ONNX model:

```
wget https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx
```

## Data

1. Download DataSet: [ILSVRC2012](https://image-net.org/) for validation.

1. Sort and label images:

Execute the following script into the validation dataset dir and get the `img.txt`.

```
import os
cur_path = os.getcwd()
file = []
for filename in os.listdir('.'):
    if filename.startswith("n"):
        file.append(filename)
file.sort()

with open("img.txt","w") as f:
    for i in range(len(file)):
        path = os.path.join(cur_path, file[i])
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_path += " "
            img_path += str(i)
            img_path += '\n'
            f.write(img_path)
```

3. Copy `img.txt` into the current dir.

## Convert

Convert and optimize the ONNX model based on PopRT.

Export converted FP32 ONNX model:

```
poprt \
--input_model googlenet-9.onnx \
--output_model googlenet_fp32.onnx
```

Export converted Fp16 ONNX model:

```
poprt \
--input_model googlenet-9.onnx \
--output_model googlenet_fp16.onnx \
--precision fp16 \
--fp16_skip_op_types LRN
```

## Run

```
# Fp32 ONNX model
python googlenet.py --model_path=googlenet_fp32.onnx

Result:
top-1-acc: 0.67778
top-5-acc: 0.88334
```

```
# Fp16 ONNX model
python googlenet.py --model_path=googlenet_fp16.onnx

Result:
top-1-acc: 0.6778
top-5-acc: 0.88328
```
