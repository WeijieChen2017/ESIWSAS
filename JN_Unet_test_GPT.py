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
        LoadImaged(keys=["image", "label"], image_only=False),
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

order_list_cnt = 32 # len(order_list)
for case_num in range(6):
    # case
    # model.eval()
    with torch.no_grad():
        # save val_ds[case_num] for further analysis

        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.from_numpy(np.expand_dims(img, 1)).float().cuda()
        val_labels = torch.from_numpy(np.expand_dims(label, 1)).float().cuda()

        _, _, ax, ay, az = val_labels.size()
        total_pixel = ax * ay * az
        output_array = np.zeros((ax, ay, az, order_list_cnt))
        for idx_bdo in range(order_list_cnt):
            # print(idx_bdo)
            # print(device)
            val_outputs = sliding_window_inference(
                val_inputs, [96, 96, 96], 8, model, overlap=1/8, device=device,
                mode="gaussian", sigma_scale=0.125, padding_mode="constant", # , order=order_list[idx_bdo],
            )
            output_array[:, :, :, idx_bdo] = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :]

        val_mode = np.asarray(np.squeeze(mode(output_array, axis=3).mode), dtype=int)

        for idx_diff in range(order_list_cnt):
            output_array[:, :, :, idx_diff] -= val_mode
        output_array = np.abs(output_array)
        output_array[output_array>0] = 1

        val_pct = np.sum(output_array, axis=3)/order_list_cnt

        np.save(
            test_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_vote.npy"), 
            val_inputs.cpu().numpy()[0, 0, :, :, :],
        )
        print(test_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            test_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_vote.npy"), 
            val_labels.cpu().numpy()[0, 0, :, :, :],
        )
        print(test_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            test_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_vote.npy"), 
            val_mode,
        )
        print(test_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            test_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"), 
            val_pct,
        )
        print(test_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"))