# SuperPoint

## Installation

[MS-COCO 2014](http://cocodataset.org/#download) and [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) should be downloaded into `$DATA_DIR`. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:
```
$DATA_DIR
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
```

## Usage
When training a model or exporting its predictions, you will often have to change the relevant configuration file in `configs/`.

### 1) Training MagicPoint on Synthetic Shapes
Run `SyntheticShapes.py`.
Change mode to ```mode = 'shapes'``` in `MagicPoint.py` and modify path outputs in `config__mp__ss.yaml` for the saved models and tensorboard summary, then run `MagicPoint.py`.

### 2) Exporting detections on MS-COCO
Change mode to ```mode = 'mp_export'``` in `Coco.py` and then run it. Adjust the model parameter in `config__mp__coco__export.yaml` to the saved model path from Step 1), and then run `export__detections.py`.
This will save the pseudo-ground truth interest point labels to `$DATA_DIR/exports/magic-point_coco_1/`.
Next, change mode to ```mode = 'mp_train'``` in `Coco.py` and adjust the `labels` and `picklefile` entries in `config__mp__coco__train.yaml` to the desired locations, and then run `Coco.py` to create a file with path mappings to the newly created labels.

### 3) Training MagicPoint on MS-COCO
Indicate the paths to the interest point labels from Step 2) in `config__mp__coco__train.yaml` by setting the entry `picklefile`. Additionally, modify path outputs for the saved models and tensorboard summary. Change mode to ```mode = 'coco'``` in `MagicPoint.py` and then run it. You might repeat steps 2) and 3) one more time.

### 4) Training of SuperPoint on MS-COCO
Once MagicPoint has been trained with a couple rounds of homographic adaptation, export again the detections on MS-COCO as in Step 2) and use these detections to train SuperPoint by setting the entry `picklefile`. Additionally, modify path outputs for the saved models and tensorboard summary.
Run `SuperPoint.py`.

## Comments
Step 3) and Step 4) have the option to utilize pretrained weights from models trained in previous steps. Just set `pretrained_model` to `True` in `config__mp_coco__train.yaml` or `config__sp_coco__train.yaml`, respectively and modify the `pretrained_weights` entry.
