# Slum expansion during COVID-19: Mapping a decade of satellite imagery using minimal labels #
This repo is the PyTorch code of our paper [Slum expansion during COVID-19: Mapping a decade of satellite imagery using minimal labels]

## Usage ##
```
usage: main.py [-h] --data-root DATA_ROOT --dataset DATASET [--batch-size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
               [--crop-size CROP_SIZE] [--backbone {resnet50,resnet101}]
               [--model {deeplabv3plusaux,deeplabv3plus,pspnet,deeplabv2}] --labeled-id-path LABELED_ID_PATH
               --unlabeled-id-path UNLABELED_ID_PATH --pseudo-mask-path PSEUDO_MASK_PATH --save-path SAVE_PATH
               [--reliable-id-path RELIABLE_ID_PATH] [--plus] [--class_name CLASS_NAME] [--labeled_num LABELED_NUM]
```
