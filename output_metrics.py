# proj_list = [
#     ["Seg532_Unet_channnel_dropoutRate_010"],
#     ["Seg532_Unet_channnel_dropoutRate_020"],
#     ["Seg532_Unet_neuron_dropoutRate_020"],
#     ["Seg532_Unet_neuron_dropoutRate_010"],
# ]

# #abbreviate the project list
# abbrev_proj_list = [
#     "channel_010",
#     "channel_020",
#     "neuron_020",
#     "neuron_010",
# ]

# img_files = [
#     "img0026",
#     "img0027",
#     "img0028",
#     "img0029",
#     "img0030",
#     "img0031",
# ]

# n_class = 14
# n_img = len(img_files)

# # load "('dice_score.xlsx')" in root directory
# import numpy as np
# import xlsxwriter
# import os

# # new excel file
# workbook_new = xlsxwriter.Workbook('dice_score_3f.xlsx')
# worksheet_new = workbook_new.add_worksheet()

# # load old excel file
# workbook_old = xlsxwriter.Reader('dice_score.xlsx')

# # load each worksheet from old excel file
# for idx_proj, proj_info in enumerate(proj_list):
#     print(idx_proj+1, proj_info[0])
#     worksheet_old = workbook_old.get_worksheet_by_name(abbrev_proj_list[idx_proj])
#     print(worksheet_old)
    
#     worksheet_new.write(0, 0, "class")
#     worksheet_new.write(0, 1, "project")
#     worksheet_new.write(0, 2, "mean")
#     worksheet_new.write(0, 3, "std")
#     worksheet_new.write(0, 4, "output")

#     for idx_class in range(n_class):
#         worksheet_new.write(idx_class+1, 0, idx_class)
#         worksheet_new.write(idx_class+1, 1, abbrev_proj_list[idx_proj])
#         curr_class_mean = worksheet_old.cell(idx_class+1, 2).value
#         curr_class_std = worksheet_old.cell(idx_class+1, 3).value
#         worksheet_new.write(idx_class+1, 2, curr_class_mean)
#         worksheet_new.write(idx_class+1, 3, curr_class_std)
#         curr_output_str = f"{1:3f}±{2:3f}".format(curr_class_mean, curr_class_std)
#         worksheet_new.write(idx_class+1, 4, curr_output_str)

#     print("Project ", abbrev_proj_list[idx_proj], " is done.")

# workbook_new.close()
# workbook_old.close()


import pandas as pd

# Define your project list and abbreviations as before
proj_list = [
    ["Seg532_Unet_channnel_dropoutRate_010"],
    ["Seg532_Unet_channnel_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_020"],
    ["Seg532_Unet_neuron_dropoutRate_010"],
]

abbrev_proj_list = [
    "channel_010",
    "channel_020",
    "neuron_020",
    "neuron_010",
]

n_class = 14

# Load the old Excel file
# Assuming 'dice_score.xlsx' is in the same directory as your script
excel_path = 'dice_score.xlsx'

# Create a new DataFrame for the processed data
processed_data = pd.DataFrame(columns=["class", "project", "mean", "std", "output"])

# write all sheet_name in the excel file
sheet_names = pd.ExcelFile(excel_path).sheet_names
print(sheet_names)

for idx_proj, proj_info in enumerate(abbrev_proj_list):
    print(f"Loading sheet: {proj_info}")
    try:
        # Load the specific worksheet
        df_old = pd.read_excel('dice_score.xlsx', sheet_name=proj_info)
        
        # Process each class in the old worksheet
        for idx_class in range(n_class):
            curr_class_mean = df_old.iloc[idx_class, 2]
            curr_class_std = df_old.iloc[idx_class, 3]
            curr_output_str = f"{curr_class_mean:.3f}±{curr_class_std:.3f}"
            
            # Append the processed data to the new DataFrame
            processed_data = processed_data.append({
                "class": idx_class,
                "project": proj_info,
                "mean": curr_class_mean,
                "std": curr_class_std,
                "output": curr_output_str
            }, ignore_index=True)
            
        print(f"Loaded sheet: {proj_info}")
    except Exception as e:
        print(f"Error loading sheet: {proj_info}, {e}")

# Save the processed data to a new Excel file
processed_data.to_excel('dice_score_3f.xlsx', index=False)
print("New Excel file has been created: dice_score_3f.xlsx")