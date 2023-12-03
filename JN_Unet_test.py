import os
# from monai.networks.nets.unet import UNet
from model import UnetMonaiNeuronDO as UNet_neuron
from model import UnetMonaiChannelDO as UNet_channel
from monai.networks.layers.factories import Act, Norm
# from utils import iter_all_order
from scipy.stats import mode

proj_list = [
    # ["Seg532_Unet_neuron_dropoutRate_020", "neuron", 0.2],
    # ["Seg532_Unet_neuron_dropoutRate_010,", "neuron", 0.1],
    ["Seg532_Unet_channnel_dropoutRate_010", "channel", 0.1],
    ["Seg532_Unet_channnel_dropoutRate_010w", "channel_w", 0.1],
    ["Seg532_Unet_channnel_dropoutRate_020", "channel", 0.2],
    # ["Seg532_Unet_channnel_dropoutRate_020w", "channel", 0.2],
]

for proj_info in proj_list:
    print(proj_info[0])

print("Project index: ", end="")
proj_idx = int(input()) - 1

n_cls = 14
test_dict = {}
test_dict["root_dir"] = "./project_dir/"+proj_list[proj_idx][0]+"/"
if not os.path.exists(test_dict["root_dir"]):
    os.mkdir(test_dict["root_dir"])
test_dict["data_dir"] = "./data_dir/JN_BTCV/"
test_dict["split_JSON"] = "dataset_532.json"
test_dict["gpu_list"] = [0]
test_dict["model_type"] = proj_list[proj_idx][1]
test_dict["dropout_rate"] = proj_list[proj_idx][2]
test_dict["eval_cnt"] = 32

import os

import numpy as np
import nibabel as nib
import torch.nn as nn

from monai.inferers import sliding_window_inference

import torch
from monai.data import (
    load_decathlon_datalist,
)

root_dir = test_dict["root_dir"]
print(root_dir)

data_dir = test_dict["data_dir"]
split_JSON = test_dict["split_JSON"]

datasets = data_dir + split_JSON
test_files = load_decathlon_datalist(datasets, True, "test")

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
with torch.no_grad():
    for idx, test_tuple in enumerate(test_files):
        img_path = test_tuple['image']
        lab_path = test_tuple['label']
        file_name = os.path.basename(lab_path)
        input_data = nib.load(img_path).get_fdata()
        lab_file = nib.load(lab_path)
        ax, ay, az = input_data.shape
        output_array = np.zeros((eval_cnt, ax, ay, az))

        # ScaleIntensityRanged(
        #     keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        # ),
        a_min=-175
        a_max=250
        b_min=0.0
        b_max=1.0
        input_data = (input_data - a_min) / (a_max - a_min)
        input_data[input_data > 1.] = 1.
        input_data[input_data < 0.] = 0.

        input_data = np.expand_dims(input_data, (0,1))
        input_data = torch.from_numpy(input_data).float().to(device)
        for idx_bdo in range(eval_cnt):
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = [96, 96, 96], 
                    sw_batch_size = 4, 
                    predictor = model,
                    overlap=0.25, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    )
            print(y_hat.shape)
            # np.save("raw_output.npy", y_hat.cpu().detach().numpy())
            # exit()
            y_hat = nn.Softmax(dim=1)(y_hat).cpu().detach().numpy()
            y_hat = np.argmax(np.squeeze(y_hat), axis=0)
            print(np.unique(y_hat))
            output_array[idx_bdo, :, :, :] = y_hat

        # val_median = np.median(output_array, axis=0)
        # val_std = np.std(output_array, axis=0)
        # use mode function to do majority vote
        val_mode = np.squeeze(mode(output_array, axis=0).mode)
        # compute the pencentage of majority vote
        val_pct = np.squeeze(mode(output_array, axis=0).count / eval_cnt)

        test_file = nib.Nifti1Image(np.squeeze(val_mode), lab_file.affine, lab_file.header)
        test_save_name = test_dict["root_dir"]+file_name.replace(".nii.gz", "_pred_seg.nii.gz")
        nib.save(test_file, test_save_name)
        print(test_save_name)

        test_file = nib.Nifti1Image(np.squeeze(val_pct), lab_file.affine, lab_file.header)
        test_save_name = test_dict["root_dir"]+file_name.replace(".nii.gz", "_pct_seg.nii.gz")
        nib.save(test_file, test_save_name)
        print(test_save_name)