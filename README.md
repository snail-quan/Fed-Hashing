# Fed-Hashing
Use hash learning and federation learning to achieve private and efficient cross-modal retrieval

## Dependencies
pytorch 1.7.1

## Training
### Processing dataset
1. Download the oringal data from coco, nuswide, mirflickr25k
2. Use the "data/make_XXX.py" to generate .mat file

Detail reference [DCHMT](https://github.com/kalenforn/DCHMT)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of CLIP/clip/clip.py. This code is based on the "ViT-B/32".

Copy ViT-B-32.pt to this dir.

### Start
```
python main.py --is-train --dataset coco --output-dim 64 --save-dir ./result/coco/64 --batch-size 256
```
