proj_list = [
    ["Seg532_Unet_channnel_dropoutRate_010"],
    ["Seg532_Unet_channnel_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_010"],
]

img_files = [
    "img0026",
    "img0027",
    "img0028",
    "img0029",
    "img0030",
    "img0031",
]

import numpy as np

for idx_proj, proj_info in enumerate(proj_list):
    
    print(idx_proj+1, proj_info[0])

    # load from img0026 to img0031 ending 
    # _y_RAS_1.5_1.5_2.0_vote.npy and _z_RAS_1.5_1.5_2.0_vote.npy
    # then compute the dice score

    for idx_img, img_name in enumerate(img_files):
        path_y = "./project_dir/" + proj_info[0] + "/" + img_name + "_y_RAS_1.5_1.5_2.0_vote.npy"
        path_z = "./project_dir/" + proj_info[0] + "/" + img_name + "_z_RAS_1.5_1.5_2.0_vote.npy"

        data_y = np.load(path_y, allow_pickle=True)
        data_z = np.load(path_z, allow_pickle=True)

        # print unique values
        print("--->", img_name, "y", np.unique(data_y))
        print("--->", img_name, "z", np.unique(data_z))
    print()