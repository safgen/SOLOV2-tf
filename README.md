# SOLOV2-tf
A Tensorflow implementation of SOLOV2 instance segmentation.
This repo is still work in progress and executing it may result in some errors.
Inference part hasn't been implemented yet.

## Installing dependencies

```bash
pip install -r requirements.txt
```

## Data Prep

To prepare the the TFRecoed dataset for the bus class, run the following from the root repo path 
```bash
cd data
# for training samples
python tfrecord_creator_coco.py --annotation path/to/instances_train2017.json --image_dir path/to/images/train2017/ --output_path ../dataset/train/train.tfrecord --num_shards <integer>
# val samples
python tfrecord_creator_coco.py --annotation path/to/instances_val2017.json --image_dir path/to/images/val2017/ --output_path ../dataset/val/val.tfrecord --num_shards <integer>
```
 
 After data peparation, run train.py with the as follows:
 ```bash
 python train.py --dataset_train ds/train/ --dataset_val ds/val/
 ```
