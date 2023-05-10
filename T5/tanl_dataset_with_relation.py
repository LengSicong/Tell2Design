from typing import DefaultDict
from PIL import Image
import numpy as np

from numpy.core.fromnumeric import shape
import utils
import os
import random
import json
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

room_idx = {0:'living room', 1:'master room', 2:'kitchen', 3:'bathroom', 4:'dining room', 5:'common room 2', 6:'common room 3', 7:'common room 1', 8:'common room 4', 9:'balcony'
            , 10:'entrance', 11:'storage'}

room_type_list = ['living room', 'master room', 'kitchen', 'bathroom', 'dining room', 'common room', 'balcony', 'entrance', 'storage']

def read_data(train_dir):
    train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    
    print(f'Number of dataset: {len(train_data_path)}')

    all_data = []
    sample_index = 0
    for path in tqdm(train_data_path):

        # restrict training size
        # if sample_index > 10254 and sample_index < 12001:
        # if sample_index > 10000 and sample_index < 12001:
        # if sample_index > 0 and sample_index < 10001:
        if sample_index > 0 and sample_index < 30:
            with Image.open(path) as temp:
                image_array = np.asarray(temp, dtype=np.uint8)
            boundary_mask = image_array[:,:,0]

            # (boundary_mask == 255).sum()
            category_mask = image_array[:,:,1]
            index_mask = image_array[:,:,2]
            inside_mask = image_array[:,:,3]
            shape_array = image_array.shape
            index_category = []
            room_node = []
            
            # Save all boundary pixels (ZY)
            boundary_pixels = (boundary_mask == 127).nonzero() # Seems not to be used 
            # To be consistent with Sicong's structure
            boundary_pixels = [(int(boundary_pixels[0][i]), int(boundary_pixels[1][i])) for i in range(len(boundary_pixels[0]))]

            # width of wall is 3 (not really)
            interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
            interiorWall_mask[category_mask == 16] = 1        
            # width of door is 3 
            interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
            interiordoor_mask[category_mask == 17] = 1
            interiordoor_pixel = interiordoor_mask.nonzero()

            # compute apartment centroid (ZY)
            inside_pixel = inside_mask.nonzero()
            apart_centroid = (inside_pixel[0].mean(), inside_pixel[1].mean())
            # # Round version
            # apart_centroid = (round(inside_pixel[0].mean()), round(inside_pixel[1].mean()))

            # Get index category pair (ZY)
            max_num_rooms = 15 # Assume the max number of rooms is 15 
            for index in range(1, max_num_rooms): # Valid index from 1
                category = category_mask[index_mask == index]
                if category.size > 0 and category[0] < 12:
                    index_category.append((index, category[0]))

            for (index, category) in index_category:
                node = {}
                # Compute box information (ZY)
                mask = index_mask == index
                centroid, sides, size, area, edge_points = utils.compute_box_info(mask)
                node['x_min'] = edge_points[0][0]
                node['y_min'] = edge_points[0][1]
                node['x_max'] = edge_points[1][0]
                node['y_max'] = edge_points[1][1]
                node['category'] = int(category)
                node['centroid'] = centroid
                node['size'] = size
                node['sides'] = sides
                node['area'] = area
                node['edge_points'] = edge_points
                node['ratio'] = utils.nearest_normal_aspect_ratio(sides[1], sides[0], maxWidth = 16, maxHeight = 16)
                node['mask'] = mask
                room_node.append(node)

            # Find nearby rooms
            all_x_min = np.array([room['x_min'] for room in room_node])
            all_y_min = np.array([room['y_min'] for room in room_node])
            all_x_max = np.array([room['x_max'] for room in room_node])
            all_y_max = np.array([room['y_max'] for room in room_node])
            side_width = 6
            for idx in range(len(room_node)):
                room = room_node[idx]
                # x_min (y_min) can only be shared by other x_max (y_max)
                # if two boxes are close enough (< 6, usually the margin is 4)
                relation = []
                dist_x_min = abs(room['x_min'] - all_x_max)
                room_node[idx]['near_x_min'] = room_node[dist_x_min.argmin()]['x_max'] if dist_x_min.min() < side_width else -1
                relation += (dist_x_min < side_width).nonzero()[0].tolist()
                dist_y_min = abs(room['y_min'] - all_y_max)
                room_node[idx]['near_y_min'] = room_node[dist_y_min.argmin()]['y_max'] if dist_y_min.min() < side_width else -1
                relation += (dist_y_min < side_width).nonzero()[0].tolist()
                dist_x_max = abs(room['x_max'] - all_x_min)
                room_node[idx]['near_x_max'] = room_node[dist_x_max.argmin()]['x_min'] if dist_x_max.min() < side_width else -1
                relation += (dist_x_max < side_width).nonzero()[0].tolist()
                dist_y_max = abs(room['y_max'] - all_y_min)
                room_node[idx]['near_y_max'] = room_node[dist_y_max.argmin()]['y_min'] if dist_y_max.min() < side_width else -1
                relation += (dist_y_max < side_width).nonzero()[0].tolist()
                room_node[idx]['relation'] = list(set(relation)) # make values unique

            # Attributes:        
            description_attri_all = DefaultDict()

            # Check whether there are multiple rooms
            isMult = {item: 0 for item in room_type_list}
            for room in room_node:
                if room['category'] in [5, 6, 7, 8]:
                    isMult["common room"] += 1
                else:
                    isMult[room_idx[room['category']]] += 1

            # distinguish multiple bathroom and balcony
            room_count = {item:0 for item in room_type_list}
            for room in room_node:
                room_type = str(room_idx[room['category']])

                # handle the same room types (ZY)
                if room['category'] in [5, 6, 7, 8]:
                    room_count["common room"] += 1
                    if isMult["common room"] > 1:
                        room_name = "common room " + str(room_count["common room"])
                    else:
                        room_name = "common room"
                else:
                    room_count[room_type] += 1
                    if isMult[room_type] > 1:
                        room_name = room_type + " " + str(room_count[room_type])
                    else:
                        room_name = room_type

                room_type = room_name

                # # For debuging
                # plt.imshow(category_mask, interpolation='nearest')
                # plt.savefig('foo.png')
            
                description = {'size':None,'aspect ratio':None,'location':None,'private':None}

                room_size = str(room['size'])
                room_ratio = room['ratio']

                # Fix bug, when size < 100 will output 0
                round_room_size = round(room['size'] / 50) * 50
                if round_room_size > 0:
                    round_room_size = str(round_room_size)
                else:
                    round_room_size = str(room['size'])
                # round_room_size = str(round(room['size'] / 100) * 100)
                round_room_ratio = str(f'{room_ratio[0]} over {room_ratio[1]}')

                description['aspect ratio'] = round_room_ratio
                description['size'] = round_room_size

                # location -> attri_5
                room_centroid = room['centroid']
                if room['category'] != 0: # living room location not considered
                    if room_centroid[0] > apart_centroid[0]: # around north
                        if abs(room_centroid[1]-apart_centroid[1]) < 40:
                            location = "north side"
                        elif room_centroid[1] < apart_centroid[1]:
                            location = "north west corner"
                        elif room_centroid[1] > apart_centroid[1]:
                            location = "north east corner"
                    else: # around south
                        if abs(room_centroid[1]-apart_centroid[1]) < 40:
                            location = "south side"
                        elif room_centroid[1] < apart_centroid[1]:
                            location = "south west corner"
                        elif room_centroid[1] > apart_centroid[1]:
                            location = "south east corner"
                    # loc_des = random.choice(location_template).format(room_type, location)
                    # description.append(loc_des)
                    description['location'] = location

                # (ZY)
                bedroom_idx = [1,5,6,7,8]
                if room['category'] in bedroom_idx:
                    max_x = room['edge_points'][1][0]
                    min_x = room['edge_points'][0][0]
                    max_y = room['edge_points'][1][1]
                    min_y = room['edge_points'][0][1]
                    description['private'] = False
                    
                    # check whether has a bathroom
                    for rm in room_node:
                        if rm['centroid'][0] in range(min_x, max_x) and rm['centroid'][1] in range(min_y, max_y) and rm['category'] == 3: # if rm is a bathroom
                            description['private'] = True

                # Check if private (ZY)
                if room['category'] in [3, 9]:
                    mask = room['mask']
                    edge_pts = room['edge_points']
                    # Find the interior door for each room 
                    room_door_mask_x = (interiordoor_pixel[0] < edge_pts[1][0] + 4) * (interiordoor_pixel[0] > edge_pts[0][0] - 4)
                    room_door_mask_y = (interiordoor_pixel[1] < edge_pts[1][1] + 4) * (interiordoor_pixel[1] > edge_pts[0][1] - 4)
                    if (room_door_mask_x*room_door_mask_y).sum() > 0: # If there is a interior door
                        # Top-left and bottom-right points of the interior door
                        room_door_pixel = (interiordoor_pixel[0][room_door_mask_x*room_door_mask_y], 
                            interiordoor_pixel[1][room_door_mask_x*room_door_mask_y])
                        room_door_pts = [(room_door_pixel[0].min(), room_door_pixel[1].min()), (room_door_pixel[0].max(), room_door_pixel[1].max())]
                        height, width = room_door_pts[1][0] - room_door_pts[0][0] + 1, room_door_pts[1][1] - room_door_pts[0][1] + 1
                        # Check if the interior door connects to the living room
                        # Within a 3x3 window
                        connect_region_tl = category_mask[room_door_pts[0][0]-1:room_door_pts[0][0]+2, room_door_pts[0][1]-1:room_door_pts[0][1]+2]
                        connect_region_br = category_mask[room_door_pts[1][0]-1:room_door_pts[1][0]+2, room_door_pts[1][1]-1:room_door_pts[1][1]+2] 

                        if 0 in connect_region_tl or 0 in connect_region_br:
                            description['private'] = False
                        else:
                            description['private'] = True

                        # description['private'] = False
                        center = (room_door_pts[0][0] + round((room_door_pts[1][0] - room_door_pts[0][0])/2),
                                room_door_pts[0][1] + round((room_door_pts[1][1] - room_door_pts[0][1])/2))
                        if height > width:
                            room1 = category_mask[center[0], center[1] - 5]
                            room2 = category_mask[center[0], center[1] + 5]
                        else:
                            room1 = category_mask[center[0] - 5, center[1]]
                            room2 = category_mask[center[0] + 5, center[1]]
                        if room1 == room['category']:
                            room_outside = room2
                        else:
                            room_outside = room1
                        if room_outside == 0:
                            description['private'] = False
                        else:
                            description['private'] = True
                    else:
                        description['private'] = False

                attribute_num_sents = len(description)

                description_attri_all[f'{room_type}'] = DefaultDict()
                description_attri_all[f'{room_type}']['description'] = description
                description_attri_all[f'{room_type}']['x'] = room['centroid'][0]
                description_attri_all[f'{room_type}']['y'] = room['centroid'][1]
                description_attri_all[f'{room_type}']['h'] = room['sides'][0]
                description_attri_all[f'{room_type}']['w'] = room['sides'][1]
                description_attri_all[f'{room_type}']['x_min'] = room['x_min']
                description_attri_all[f'{room_type}']['y_min'] = room['y_min']
                description_attri_all[f'{room_type}']['x_max'] = room['x_max']
                description_attri_all[f'{room_type}']['y_max'] = room['y_max']
                description_attri_all[f'{room_type}']['near_x_min'] = room['near_x_min']
                description_attri_all[f'{room_type}']['near_y_min'] = room['near_y_min']
                description_attri_all[f'{room_type}']['near_x_max'] = room['near_x_max']
                description_attri_all[f'{room_type}']['near_y_max'] = room['near_y_max']
                description_attri_all[f'{room_type}']['relation'] = room['relation']
                description_attri_all[f'{room_type}']['num_attributes'] = attribute_num_sents
                # if room_type == 'living room':
                #     description_attri_all[f'{room_type}']['description'] = description[:attribute_num_sents]

            sample = DefaultDict()
            tokens = []
            rooms = []
            start_index = 0
            for room in description_attri_all:
                room_info = DefaultDict()
                room_info['room_type'] = room
                room_info['x'] = description_attri_all[room]['x']
                room_info['y'] = description_attri_all[room]['y']
                room_info['h'] = description_attri_all[room]['h']
                room_info['w'] = description_attri_all[room]['w']
                room_info['x_min'] = description_attri_all[room]['x_min']
                room_info['y_min'] = description_attri_all[room]['y_min']
                room_info['x_max'] = description_attri_all[room]['x_max']
                room_info['y_max'] = description_attri_all[room]['y_max']
                room_info['near_x_min'] = description_attri_all[room]['near_x_min']
                room_info['near_y_min'] = description_attri_all[room]['near_y_min']
                room_info['near_x_max'] = description_attri_all[room]['near_x_max']
                room_info['near_y_max'] = description_attri_all[room]['near_y_max']

                room_names = list(description_attri_all.keys())
                room_info['relation'] = [room_names[i] for i in description_attri_all[room]['relation']]
                room_info['location'] = description_attri_all[room]['description']['location']
                room_info['size'] = description_attri_all[room]['description']['size']
                room_info['aspect ratio'] = description_attri_all[room]['description']['aspect ratio']
                # only bathroom, balcony and bedroom has True/False <private> key value. Other rooms have value None.
                room_info['private'] = description_attri_all[room]['description']['private']

                rooms.append(room_info)

            sample['rooms'] = rooms
            sample['boundary'] = boundary_pixels
            sample["boundary_boxs"] = boundary2box(smooth_boundary(boundary_mask))
            sample_index += 1
            all_data.append(sample)
        else:
            sample_index += 1
    with open('./data/floorplan/floorplan_train.json','w') as fp:
    # with open('./data/floorplan/floorplan_dev.json','w') as fp:
        json.dump(all_data, fp)    
            

def boundary2box(boundary_mask):
    ######### Start transform outline to boxes #########
    door_mask = boundary_mask == 255
    boundary_mask[door_mask] = 127

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(boundary_mask, 27, 0.01, 10)
    corners = np.int0(corners)

    # Find the largest bbox
    max_x = max(corners[:,:,1])[0]
    min_x = min(corners[:,:,1])[0]
    max_y = max(corners[:,:,0])[0]
    min_y = min(corners[:,:,0])[0]

    # (room_type = positive, bbox_x, bbox_y, bbox_h, bbox_w)
    bbox_h = max_x - min_x
    bbox_w = max_y - min_y
    bbox_x = min_x + round(bbox_h/2)
    bbox_y = min_y + round(bbox_w/2)

    boundary_box = {}
    boundary_box["room_type"] = "positive"
    boundary_box["x"] = int(bbox_x)
    boundary_box["y"] = int(bbox_y)
    boundary_box["h"] = int(bbox_h)
    boundary_box["w"] = int(bbox_w)
    boundary_box["x_min"] = int(min_x)
    boundary_box["y_min"] = int(min_y)
    boundary_box["x_max"] = int(max_x)
    boundary_box["y_max"] = int(max_y)

    boundary_boxes = []
    boundary_boxes.append(boundary_box)

    internal_ptr = []
    for i in corners:
        y, x = i.ravel() 
        # Find points inside the outline
        if x in range(min_x+1, max_x) and y in range(min_y+1, max_y):
            internal_ptr.append((x, y))

    # Find the end point from each internal point 
    for (x, y) in internal_ptr:
        h, w = 0, 0
        edge_value = boundary_mask[x, y]
        if boundary_mask[x + 1, y] == edge_value:
            x_step = 1
        elif boundary_mask[x - 1, y] == edge_value:
            x_step = -1
        else:
            raise Exception('Unexpected point!')

        for i in range(boundary_mask.shape[0]):
            if boundary_mask[x + i*x_step, y] != 0:
                h += 1
            else:
                break
        if boundary_mask[x, y + 1] == edge_value:
            y_step = 1
        elif boundary_mask[x, y - 1] == edge_value:
            y_step = -1
        else:
            raise Exception('Unexpected point!')
        for i in range(boundary_mask.shape[1]):
            if boundary_mask[x, y + i*y_step] != 0:
                w += 1
            else:
                break
        end_ptr = (x + x_step*h, y + y_step*w)

        # Handle boxes with the end point is inside the boundary
        if end_ptr[0] in range(min_x+1, max_x) and end_ptr[1] in range(min_y+1, max_y):
            boundary_box = {} 
            if x_step > 0:
                h = x - min_x
            else:
                h = max_x - x
            if y_step > 0:
                w = y - min_y
            else:
                w = max_y - y
            boundary_box["room_type"] = "negative"
            boundary_box["x"] = round(x - x_step*h/2)
            boundary_box["y"] = round(y - y_step*w/2)
            boundary_box["h"] = int(h)
            boundary_box["w"] = int(w)
        else:
            boundary_box = {}
            boundary_box["room_type"] = "negative"
            boundary_box["x"] = round(x + x_step*h/2)
            boundary_box["y"] = round(y + y_step*w/2)
            boundary_box["h"] = int(h)
            boundary_box["w"] = int(w)
        boundary_box["x_min"] = round(boundary_box["x"] - h/2)
        boundary_box["y_min"] = round(boundary_box["y"] - h/2)
        boundary_box["x_max"] = round(boundary_box["x"] + w/2)
        boundary_box["y_max"] = round(boundary_box["y"] - w/2)
        boundary_boxes.append(boundary_box)

    def draw_bbox(img, boundary_box, colcor, thickness = 3):
        thickness = round(thickness/2)
        min_x = boundary_box["x"] - round(boundary_box["h"]/2)
        max_x = boundary_box["x"] + round(boundary_box["h"]/2)
        min_y = boundary_box["y"] - round(boundary_box["w"]/2)
        max_y = boundary_box["y"] + round(boundary_box["w"]/2)

        img[min_x:max_x, min_y - thickness:min_y + thickness] = colcor
        img[min_x:max_x, max_y - thickness:max_y + thickness] = colcor

        img[min_x - thickness:min_x + thickness, min_y:max_y] = colcor
        img[max_x - thickness:max_x + thickness, min_y:max_y] = colcor

        return img

    # # For Debuging
    # for box in boundary_boxes:
    #     draw_bbox(boundary_mask, box, 255)
    # plt.imshow(boundary_mask), plt.savefig('foo.png')

    return boundary_boxes
    ######### End transform outline to boxes #########

def smooth_boundary(boundary_mask):
    door_mask = boundary_mask == 255
    boundary_mask = boundary_mask.copy()
    boundary_mask[door_mask] = 127

    do_smooth = True
    while do_smooth:
        boundary_temp = boundary_mask > 0
        num_pixel_x = boundary_temp.sum(1)
        num_pixel_y = boundary_temp.sum(0)
        new_boundary_mask = boundary_mask.copy()

        num_invalid = 0
        for i in range(1, boundary_mask.shape[0] - 1):
            if num_pixel_x[i] != num_pixel_x[i+1] and num_pixel_x[i] != num_pixel_x[i-1]:
                num_invalid += 1
            if num_pixel_y[i] != num_pixel_y[i+1] and num_pixel_y[i] != num_pixel_y[i-1]:
                num_invalid += 1
        if num_invalid == 0:
            break
        else:
            num_invalid

        # Find invalid rows
        for i in range(1, boundary_mask.shape[0] - 1):
            if num_pixel_x[i] != num_pixel_x[i+1] and num_pixel_x[i] != num_pixel_x[i-1]:
                if num_pixel_x[i-1] < num_pixel_x[i+1]:
                    # copy_idx = i-1
                    copy_idx = i+1
                else:
                    # copy_idx = i+1
                    copy_idx = i-1
                # new_boundary_mask[i] = boundary_mask[copy_idx]
                new_boundary_mask[i] = new_boundary_mask[copy_idx]

            if num_pixel_y[i] != num_pixel_y[i+1] and num_pixel_y[i] != num_pixel_y[i-1]:
                if num_pixel_y[i-1] < num_pixel_y[i+1]:
                    # copy_idx = i-1
                    copy_idx = i+1
                else:
                    # copy_idx = i+1
                    copy_idx = i-1
                # new_boundary_mask[:, i] = boundary_mask[:, copy_idx]
                new_boundary_mask[:, i] = new_boundary_mask[:, copy_idx]
        boundary_mask = new_boundary_mask
    return new_boundary_mask

# read_data('./floorplan_dataset/')

if __name__=='__main__':
    # read_data('./new_baseline/create_dataset/floorplan_dataset/')
    read_data('./create_dataset/floorplan_dataset/')