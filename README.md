# Fed-Hashing
the code of Research on Information Content Retrieval Method Based on  Cross-modal Federated Hashing

## Dependencies
pytorch 1.7.1
sklearn
tqdm
pillow

## Training
### Processing dataset
Before training, you need to download the oringal data from coco(include 2017 train,val and annotations), nuswide(include all), mirflickr25k(include mirflickr25k and mirflickr25k_annotations_v080), then use the "data/make_XXX.py" to generate .mat file

For example:
> cd COCO_DIR # include train val images and annotations files
> 
> mkdir mat
> 
> cp DCMHT/data/make_coco.py mat
> 
> python make_coco.py --coco-dir ../ --save-dir ./

After all mat file generated, the dir of dataset will like this:

```
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
```
### Download CLIP pretrained model
Download CLIP pretrained model
Pretrained model will be found in the 30 lines of CLIP/clip/clip.py. This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start
```
python main.py --is-train --dataset coco --output-dim 64 --save-dir ./result/coco/64 --batch-size 256
```
