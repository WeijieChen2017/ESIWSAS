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

n_class = 14
n_img = len(img_files)

import numpy as np
from sklearn.metrics import confusion_matrix
# write to excel file
import xlsxwriter

# new excel file
workbook = xlsxwriter.Workbook('dice_score.xlsx')

for idx_proj, proj_info in enumerate(proj_list):
    
    print(idx_proj+1, proj_info[0])

    # load from img0026 to img0031 ending 
    # _y_RAS_1.5_1.5_2.0_vote.npy and _z_RAS_1.5_1.5_2.0_vote.npy
    # then compute the dice score
    dice_proj = np.zeros((n_img, n_class))

    for idx_img, img_name in enumerate(img_files):
        path_y = "./project_dir/" + proj_info[0] + "/" + img_name + "_y_RAS_1.5_1.5_2.0_vote.npy"
        path_z = "./project_dir/" + proj_info[0] + "/" + img_name + "_z_RAS_1.5_1.5_2.0_vote.npy"

        data_y = np.load(path_y, allow_pickle=True)
        data_z = np.load(path_z, allow_pickle=True)

        # # print datashape
        # print("--->", img_name, "y", data_y.shape)
        # print("--->", img_name, "z", data_z.shape)

        # # print unique values
        # print("--->", img_name, "y", np.unique(data_y))
        # print("--->", img_name, "z", np.unique(data_z))

        # convert both y and z to int
        data_y = data_y.astype(int)
        data_z = data_z.astype(int)

        # compute confusion matrix
        cm = confusion_matrix(data_y.flatten(), data_z.flatten())

        # compute the dice coefficient for each class
        dice = np.zeros((cm.shape[0],))
        for i in range(cm.shape[0]):
            dice[i] = 2*cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
            dice_proj[idx_img, i] = dice[i]
            # print("Dice score for class", i, "in", img_name, "is", dice[i])
    
    # print the mean and std dice score for the project on each class
    mean_dice = np.mean(dice_proj, axis=0)
    std_dice = np.std(dice_proj, axis=0)
    for i in range(n_class):
        print("Dice score for class", i, "is", mean_dice[i], "+/-", std_dice[i])
        # write to txt file in the folder of the project
        with open("./project_dir/" + proj_info[0] + "/dice_score.txt", "w") as f:
            f.write("Dice score for class " + str(i) + " is " + str(mean_dice[i]) + "+/-" + str(std_dice[i]) + "\n")
    
    # write to excel file, each project is a sheet
    worksheet = workbook.add_worksheet(proj_info[0])
    worksheet.write(0, 0, "Class")
    worksheet.write(0, 1, "Mean Dice")
    worksheet.write(0, 2, "Std Dice")
    for i in range(n_class):
        worksheet.write(i+1, 0, i)
        worksheet.write(i+1, 1, mean_dice[i])
        worksheet.write(i+1, 2, std_dice[i])
    
    print()

workbook.close()