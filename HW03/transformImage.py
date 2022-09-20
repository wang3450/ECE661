from copy import deepcopy

import numpy as np
from copy import *
from tqdm import tqdm

def transformInputImage(source_image, H):
    input_image = np.transpose(source_image, (1, 0, 2))
    new_image_shape_width = input_image.shape[0]
    new_image_shape_height = input_image.shape[1]
    grid = np.array(np.meshgrid(np.arange(new_image_shape_width), np.arange(new_image_shape_height), np.arange(1, 2)))
    combinations = np.transpose(grid.T.reshape(-1, 3))
    homography_inverse = np.linalg.inv(H)
    new_homogenous_coord = homography_inverse.dot(combinations)
    new_homogenous_coord = new_homogenous_coord / new_homogenous_coord[2]

    min_values = np.amin(new_homogenous_coord, axis=1)
    min_values = np.round(min_values).astype(int)
    max_values = np.amax(new_homogenous_coord, axis=1)
    max_values = np.round(max_values).astype(int)
    required_image_size = max_values - min_values
    required_image_width = int(required_image_size[0])
    required_image_height = int(required_image_size[1])

    aspect_ratio = required_image_width / required_image_height
    scaled_height = 3000
    scaled_width = scaled_height * aspect_ratio
    scaled_width = int(round(scaled_width))
    s = required_image_height / scaled_height
    # Initialize the new frame
    img_replication = np.zeros((scaled_width, scaled_height, 3), np.uint8) * 255
    # Look at all pixels in the new frame: does the homography map them into the original image frame?
    x_values_in_target = np.arange(scaled_width)
    y_values_in_target = np.arange(scaled_height)
    target_grid = np.array(np.meshgrid(x_values_in_target, y_values_in_target, np.arange(1, 2)))
    combinations_in_target = np.transpose(target_grid.T.reshape(-1, 3))
    target_combinations_wo_offset = deepcopy(combinations_in_target)
    combinations_in_target[0] = combinations_in_target[0] * s + min_values[0]
    combinations_in_target[1] = combinations_in_target[1] * s + min_values[1]
    target_new_homogenous_coord = H.dot(combinations_in_target)
    target_new_homogenous_coord = target_new_homogenous_coord / target_new_homogenous_coord[2]

    x_coordinates_rounded = np.round(target_new_homogenous_coord[0]).astype(int)
    y_coordinates_rounded = np.round(target_new_homogenous_coord[1]).astype(int)
    # Check conditions on x
    x_coordinates_greater_zero = x_coordinates_rounded > 0
    x_coordinates_smaller_x_shape = x_coordinates_rounded < new_image_shape_width
    x_coordinates_valid = x_coordinates_greater_zero * x_coordinates_smaller_x_shape
    # Check conditions on y
    y_coordinates_greater_zero = y_coordinates_rounded > 0
    y_coordinates_smaller_y_shape = y_coordinates_rounded < new_image_shape_height
    y_coordinates_valid = y_coordinates_greater_zero * y_coordinates_smaller_y_shape

    valid_coordinates = x_coordinates_valid * y_coordinates_valid
    target_valid_x = target_combinations_wo_offset[0][valid_coordinates == True]
    target_valid_y = target_combinations_wo_offset[1][valid_coordinates == True]

    list_of_x_values_target = list(target_valid_x)
    list_of_y_values_target = list(target_valid_y)
    list_of_valid_coordinate_pairs = list()

    for i in tqdm(range(len(list_of_x_values_target))):
        list_of_valid_coordinate_pairs.append([list_of_x_values_target[i], list_of_y_values_target[i]])
    valid_x = x_coordinates_rounded[valid_coordinates == True]
    valid_y = y_coordinates_rounded[valid_coordinates == True]

    list_of_x_values = list(valid_x)
    list_of_y_values = list(valid_y)

    list_of_original_coords_for_mapping = list()
    for i in tqdm(range(len(list_of_x_values))):
        list_of_original_coords_for_mapping.append([list_of_x_values[i], list_of_y_values[i]])

    j = 0
    for pair in tqdm(list_of_valid_coordinate_pairs):
        img_replication[pair[0], pair[1], :] = input_image[list_of_original_coords_for_mapping[j][0],
                                               list_of_original_coords_for_mapping[j][1], :]
        j = j + 1

    img_replication = np.transpose(img_replication, (1, 0, 2))
    return img_replication
