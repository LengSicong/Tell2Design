# %%
import os
import cv2
from tqdm import tqdm
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import jaccard_score
from collections import defaultdict

# %%
from calculate_iou import cluster_pixel_values, generate_mask, generate_one_dim_mask, find_central_point, calculate_iou

# %%
# define rooms types and their corresponding colors
last_5k_type = ['255,255,255', '0,0,0', '0,0,255', '170,232,238', '128,128,240', '230,216,173', '0,215,255', '0,165,255', '35,142,107', '221,160,221', '0,255,255', '214,112,218']
last_dict = dict.fromkeys(last_5k_type)
for key in last_dict.keys():
    last_dict[key] = [int(val) for val in key.split(',')]

# %%
# load all images
dir_paths = []
for root, dirs, files in os.walk('/home/sicong/CogView/samples_text2image/eval_set_human_only/'):
    dir_paths.extend(dirs)


# %%
# calculate IoU
Macro_IoUs = []
Micro_IoUs = []
for img_id in tqdm(dir_paths):
    gt_image = cv2.imread(os.path.join("/home/sicong/imagen-sicong/dataset/imgs/",f"{img_id}.png"))
    pred_image = cv2.imread(os.path.join(f"/home/sicong/CogView/samples_text2image/eval_set_human_only/{img_id}/","0.png"))

    new_image = cluster_pixel_values(pred_image, last_dict)

    dict_pred_mask = generate_mask(new_image, last_dict)
    dict_gt_mask = generate_mask(gt_image, last_dict)

    one_dim_pred_mask = generate_one_dim_mask(dict_pred_mask)
    one_dim_gt_mask = generate_one_dim_mask(dict_gt_mask)

    rooms = ['170,232,238', '128,128,240', '230,216,173', '0,215,255', '0,165,255', '35,142,107', '221,160,221', '0,255,255', '214,112,218']
    IoUs, max_key = find_central_point(one_dim_gt_mask, one_dim_pred_mask)
    macro_iou, micro_iou = calculate_iou(rooms, dict_gt_mask, dict_pred_mask, max_key)
    
    Macro_IoUs.append(macro_iou)
    Micro_IoUs.append(micro_iou)

# %%
print(f"Macro IoU: {np.mean(Macro_IoUs)}")
print(f"Micro IoU: {np.mean(Micro_IoUs)}")

# %%



