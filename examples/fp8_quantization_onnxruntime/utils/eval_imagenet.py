# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k.

    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_imagenet_from_directory(
    directory: str,
    subset: int = None,
    batchsize: int = 32,
    shuffle: bool = False,
    require_label: bool = True,
    num_of_workers: int = 12,
) -> torch.utils.data.DataLoader:
    dataset = datasets.ImageFolder(
        directory,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    if subset:
        dataset = Subset(dataset, indices=[_ for _ in range(0, subset)])
    if require_label:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            drop_last=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            collate_fn=lambda x: torch.cat(
                [sample[0].unsqueeze(0) for sample in x], dim=0
            ),
            drop_last=False,
        )


def evaluate_ipu_with_imagenet(
    inference,
    imagenet_validation_dir: str,
    batchsize: int = 32,
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = False,
    subset: int = None,
):
    # dataloader
    if imagenet_validation_loader is None:
        imagenet_validation_loader = load_imagenet_from_directory(
            imagenet_validation_dir, batchsize=batchsize, subset=subset, shuffle=False
        )
    recorder = {'top1_accuracy': []}

    # inference
    model_inputs, model_outputs = inference.get_io_info()
    model_inputs_name = list(model_inputs.keys())[0]
    model_outputs_name = list(model_outputs.keys())[0]
    for batch_idx, (batch_input, batch_label) in tqdm(
        enumerate(imagenet_validation_loader),
        desc='Evaluating Model...',
        total=len(imagenet_validation_loader),
    ):
        batch_input = {model_inputs_name: batch_input}
        batch_pred = inference.predict(batch_input)[model_outputs_name]
        # calculation topk
        prec1 = accuracy(
            torch.tensor(batch_pred, dtype=torch.float).to('cpu'),
            batch_label.to('cpu'),
            topk=(1,),
        )[0]
        recorder['top1_accuracy'].append(prec1.item())
        if batch_idx % 1000 == 0 and verbose:
            top1 = sum(recorder['top1_accuracy']) / len(recorder['top1_accuracy'])
            print(top1)
    return sum(recorder['top1_accuracy']) / len(recorder['top1_accuracy'])
