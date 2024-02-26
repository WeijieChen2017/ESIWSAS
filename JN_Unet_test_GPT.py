import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.networks.layers.factories import Act, Norm
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from scipy.stats import mode

# Your existing project setup and model imports remain unchanged
# from monai.networks.nets.unet import UNet
from model import UnetMonaiNeuronDO as UNet_neuron
from model import UnetMonaiChannelDO as UNet_channel
# from utils import iter_all_order

proj_list = [
    ["Seg532_Unet_channnel_dropoutRate_010", "channel", 0.1],
    ["Seg532_Unet_channnel_dropoutRate_010w", "channel_w", 0.1],
    ["Seg532_Unet_channnel_dropoutRate_020", "channel", 0.2],
]

for proj_info in proj_list:
    print(proj_info[0])

print("Project index: ", end="")
proj_idx = int(input()) - 1

test_dict = {
    "root_dir": "./project_dir/" + proj_list[proj_idx][0] + "/",
    "data_dir": "./data_dir/JN_BTCV/",
    "split_JSON": "dataset_532.json",
    "gpu_list": [0],
    "model_type": proj_list[proj_idx][1],
    "dropout_rate": proj_list[proj_idx][2],
    "eval_cnt": 32,
}

if not os.path.exists(test_dict["root_dir"]):
    os.mkdir(test_dict["root_dir"])

# Setup for MONAI transforms, CacheDataset, and DataLoader
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

datasets = test_dict["data_dir"] + test_dict["split_JSON"]
val_files = load_decathlon_datalist(datasets, True, "test")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

# GPU setup remains unchanged
gpu_list = ','.join(str(x) for x in test_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if test_dict["model_type"] == "neuron":

    model = UNet_neuron( 
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=6,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=test_dict["dropout_rate"],
        bias=True,
        ).to(device)
    
elif test_dict["model_type"] == "channel":

    model = UNet_channel( 
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=6,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        neuron_dropout=0., # neuron dropout rate
        bias=True,
        is_WDO=False,
        channel_dropout=test_dict["dropout_rate"], # channel dropout rate
        ).to(device)

elif test_dict["model_type"] == "channel_w":

    model = UNet_channel( 
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=6,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        neuron_dropout=0., # neuron dropout rate
        bias=True,
        is_WDO=True,
        channel_dropout=test_dict["dropout_rate"], # channel dropout rate
        ).to(device)

pre_train_state = {}
pre_train_model = torch.load(test_dict["root_dir"]+"best_metric_model.pth")

for model_key in model.state_dict().keys():
    pre_train_state[model_key] = pre_train_model[model_key]
     
model.load_state_dict(pre_train_state)

model.train()
eval_cnt = test_dict["eval_cnt"]


# Model inference adjustment for using DataLoader
with torch.no_grad():
    for batch_data in val_loader:
        img = batch_data["image"].to(device)
        # Assuming the original filename is included in your dataset under 'image_meta_dict'
        original_filenames = batch_data["image_meta_dict"]["filename_or_obj"]
        print(original_filenames)