proj_list = [
    ["Seg532_Unet_channnel_dropoutRate_010"],
    ["Seg532_Unet_channnel_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_010"],
]

import numpy as np

for idx, proj_info in enumerate(proj_list):
    
    print(idx+1, proj_info[0])

    # load from img0026 to img0031 ending 
    # _y_RAS_1.5_1.5_2.0_vote.npy and _z_RAS_1.5_1.5_2.0_vote.npy
    # then compute the dice score

    path_y = "./project_dir/" + proj_list[idx][0] + "_y_RAS_1.5_1.5_2.0_vote.npy"
    path_z = "./project_dir/" + proj_list[idx][0] + "_z_RAS_1.5_1.5_2.0_vote.npy"

    data_y = np.load(path_y, allow_pickle=True)
    data_z = np.load(path_z, allow_pickle=True)

    print(data_y.shape, data_z.shape)

    print()