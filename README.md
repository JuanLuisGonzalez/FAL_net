# FAL_net
This is a fork of the repository for the NeurIPS2020 Accepted paper "Forget About the LiDAR: Self-Supervised Depth Estimators with MED Probability Volumes"
Paper and supplemental materials: https://proceedings.neurips.cc/paper/2020/hash/951124d4a093eeae83d9726a20295498-Abstract.html

## Pre-trained models
Pre-trained model (KITTI-only, [1st stage](https://drive.google.com/drive/folders/1oQwfzEHz6MwDXqkc6riROYEOynyaEE-A?usp=sharing))

Pre-trained model (KITTI-only, [2nd stage](https://drive.google.com/drive/folders/1OakYov5-TQ3koiHV-i4xvuy0f8WIaOoM?usp=sharing))

## Computed inverse depth maps
Improved Eigen test split [Trained on KITTI-only, 2nd_stage, and post-processing](https://drive.google.com/drive/folders/1eOoTqefLh7tc6YiK1Kb8fjc9Ezxgte_J?usp=sharing)

## KITTI dataset
In order to reproduce the results, you will need to download the raw KITTI dataset. You can do this using the ["raw dataset download script"](http://www.cvlibs.net/datasets/kitti/raw_data.php) provided by Omid Hosseini. After downloading, you'll need to rearrange the folders into the following structure: 
```
KITTI/year_month_day_drive_number_sync/image_00/data/0000000001.png
```


Furthermore you will need to download some of the files from the [cvlibs depth challenge](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php), called data_depth_velodyne.zip and data_depth_annotated.zip. Download and extract the contents and merge them into the main KITTI dataset. On MacOS, you can this by pressing the option key while dragging and dropping the folders from the depth challenge and selecting the merge option in the prompt.


## Usefull tensorboard call

```
tensorboard --logdir=C:ProjectDir/NeurIPS2020_FAL_net/Kitti --port=6012
```

## Cite our paper:
```
@inproceedings{NEURIPS2020_951124d4,
 author = {Gonzalez Bello, Juan Luis and Kim, Munchurl},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {12626--12637},
 publisher = {Curran Associates, Inc.},
 title = {Forget About the LiDAR: Self-Supervised Depth Estimators with MED Probability Volumes},
 url = {https://proceedings.neurips.cc/paper/2020/file/951124d4a093eeae83d9726a20295498-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

New: Added files to train our FAL-net on KITTI

Coming soon: Files to train our FAL-net on KITTI + CityScapes

*Not for commercial use*
If wanted for commercial use contact juanluisgb@kaist.ac.kr
