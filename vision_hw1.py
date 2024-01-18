# per channel histogram and 3d histogram
# Grid based search and whole image search
# RGB and HSV

import numpy as np
from PIL import Image
import os

# give dataset path 
dataset_path = "dataset/"

# take image names from InstanceNames.txt
file = open(dataset_path + "/InstanceNames.txt", "r")
image_names = []
for line in file:
    image_names.append(line.strip())
file.close()

# take images as numpy array
# hold images as dictionary key: image name, value: image as numpy array
query1_images = {}
query2_images = {}
query3_images = {}
support_images = {}

query1_images_hsv = {}
query2_images_hsv = {}
query3_images_hsv = {}
support_images_hsv = {}

query1_path = dataset_path + "query_1/"
query2_path = dataset_path + "query_2/"
query3_path = dataset_path + "query_3/"
support_path = dataset_path + "support_96/"

# Convert rgb to hsv
def convert_rgb_to_hsv(image):
    normalized_image = image / 255.0
    hsv_image = np.zeros(image.shape)
    
    for row in range(normalized_image.shape[0]):
        for column in range(normalized_image.shape[1]): 
            r,g,b = normalized_image[row,column]
            C_min = min(r,g,b)
            C_max = max(r,g,b)
            delta = C_max - C_min

            # calculate hue
            if delta == 0:
                hue = 0
            elif C_max == r:
                hue = (1/6) * (((g - b) / delta) % 6)
            elif C_max == g:
                hue = (1/6) * (((b - r) / delta) + 2)
            elif C_max == b:
                hue = (1/6) * (((r - g) / delta) + 4)

            # calculate saturation
            if C_max == 0:
                saturation = 0
            elif C_max > 0:
                saturation = delta / C_max
            # calculate value
            value = C_max

            hsv_image[row,column] = [hue, saturation, value]
    
    hsv_image =  hsv_image * 255.0
    return hsv_image

for line in os.listdir(query1_path):
    image = Image.open(query1_path + line)
    image_array = np.asarray(image)
    query1_images["query_1/" + line] = image_array
    query1_images_hsv["query_1/" + line] = convert_rgb_to_hsv(np.asarray(image))

for line in os.listdir(query2_path):
    image = Image.open(query2_path + line)
    image_array = np.asarray(image)
    query2_images["query_2/" + line] = image_array
    query2_images_hsv["query_2/" + line] = convert_rgb_to_hsv(np.asarray(image))

for line in os.listdir(query3_path):
    image = Image.open(query3_path + line)
    image_array = np.asarray(image)
    query3_images["query_3/" + line] = image_array
    query3_images_hsv["query_3/" + line] = convert_rgb_to_hsv(np.asarray(image))

for line in os.listdir(support_path):
    image = Image.open(support_path + line)
    image_array = np.asarray(image)
    support_images["support/" + line] = image_array
    support_images_hsv["support/" + line] = convert_rgb_to_hsv(np.asarray(image))


# query1_images = keys: query_<no> + / + image name, values: image as numpy array
def get_per_channel_histogram(image, bin_count):
    red_channel = image[:,:,0]
    green_channel = image[:,:,1]
    blue_channel = image[:,:,2]
    
    bin_size = int(256/bin_count)
    
    red_histogram = np.zeros(bin_count, dtype=int)
    green_histogram = np.zeros(bin_count, dtype=int)
    blue_histogram = np.zeros(bin_count, dtype=int)
    
    for i in range(red_channel.shape[0]):
        for j in range(red_channel.shape[1]):
            red_histogram[int(red_channel[i,j]/bin_size)] += 1
            green_histogram[int(green_channel[i,j]/bin_size)] += 1
            blue_histogram[int(blue_channel[i,j]/bin_size)] += 1
    
    
    return np.array([red_histogram, green_histogram, blue_histogram])

def get_3D_histogram(image, bin_count):
    red_channel = image[:,:,0]
    green_channel = image[:,:,1]
    blue_channel = image[:,:,2]

    bin_size = int(256/bin_count)
    
    histogram = np.zeros((bin_count, bin_count, bin_count), dtype=int)
    
    for i in range(red_channel.shape[0]):
        for j in range(red_channel.shape[1]):
            histogram[int(red_channel[i,j]/bin_size), int(green_channel[i,j]/bin_size), int(blue_channel[i,j]/bin_size)] += 1
    histogram = histogram.flatten()
    return histogram 

# query1_histograms = keys: query_<no> + / + image name, values: per_channel histogram of image
def get_per_channel_histogram_for_each_queryset(bin_count, isRGB):
    query1_histograms = {}
    query2_histograms = {}
    query3_histograms = {}
    support_histograms = {}

    for keys in query1_images.keys():
        if(isRGB):
            query1_histograms[keys] = get_per_channel_histogram(query1_images[keys], bin_count)
        else:
            query1_histograms[keys] = get_per_channel_histogram(query1_images_hsv[keys], bin_count)

    for keys in query2_images.keys():
        if(isRGB):
            query2_histograms[keys] = get_per_channel_histogram(query2_images[keys], bin_count)
        else:
            query2_histograms[keys] = get_per_channel_histogram(query2_images_hsv[keys], bin_count)

    for keys in query3_images.keys():
        if(isRGB):
            query3_histograms[keys] = get_per_channel_histogram(query3_images[keys], bin_count)
        else:
            query3_histograms[keys] = get_per_channel_histogram(query3_images_hsv[keys], bin_count)

    for keys in support_images.keys():
        if(isRGB):
            support_histograms[keys] = get_per_channel_histogram(support_images[keys], bin_count)
        else:
            support_histograms[keys] = get_per_channel_histogram(support_images_hsv[keys], bin_count)
    
    return query1_histograms, query2_histograms, query3_histograms, support_histograms    


def get_3D_histogram_for_each_queryset(bin_count, isRGB):
    query1_histograms = {}
    query2_histograms = {}
    query3_histograms = {}
    support_histograms = {}

    for keys in query1_images.keys():
        if(isRGB):
            query1_histograms[keys] = get_3D_histogram(query1_images[keys], bin_count)
        else:
            query1_histograms[keys] = get_3D_histogram(query1_images_hsv[keys], bin_count)

    for keys in query2_images.keys():
        if(isRGB):
            query2_histograms[keys] = get_3D_histogram(query2_images[keys], bin_count)
        else:
            query2_histograms[keys] = get_3D_histogram(query2_images_hsv[keys], bin_count)

    for keys in query3_images.keys():
        if(isRGB):
            query3_histograms[keys] = get_3D_histogram(query3_images[keys], bin_count)
        else:
            query3_histograms[keys] = get_3D_histogram(query3_images_hsv[keys], bin_count)

    for keys in support_images.keys():
        if(isRGB):
            support_histograms[keys] = get_3D_histogram(support_images[keys], bin_count)
        else:
            support_histograms[keys] = get_3D_histogram(support_images_hsv[keys], bin_count)
    
    return query1_histograms, query2_histograms, query3_histograms, support_histograms

def l1_normalize_histogram(histogram):
    norm = np.sum(histogram)
    return histogram / norm  

def histogram_intersection(hist1, hist2):
    intersection = np.minimum(hist1, hist2)
    return np.sum(intersection)

def compute_image_similarity(chosen_query_histograms, support_histograms):
    similarities = {}

    for query_image, query_hist in chosen_query_histograms.items():
        for support_image, support_hist in support_histograms.items():
            # Compute the histogram intersection between query and support histograms
            query_hist_normalized = l1_normalize_histogram(query_hist)
            support_hist_normalized = l1_normalize_histogram(support_hist)
            intersection = histogram_intersection(query_hist_normalized, support_hist_normalized)
            similarities[(query_image, support_image)] = intersection

    return similarities

def compute_top1_accuracy(query_set, support_set, similarities):
    correct_matches = 0
    total_queries = len(query_set)

    for query_image in query_set:
        # Find the most similar support image for the query image
        max_similarity = -1  
        most_similar_support = None

        for support_image in support_set:
            similarity = similarities[(query_image, support_image)]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_support = support_image
        
        # Check if the filenames of the query and support images match
        if query_image.split("/")[-1] == most_similar_support.split("/")[-1]:
            correct_matches += 1

    accuracy_top1 = correct_matches / total_queries
    return accuracy_top1

# Calling function for 3D RGB Histograms
def pipeline_for_3D_RGB(bin_counts):
    # Perform experiments for 3D color histograms with RGB values
    for bin_count in bin_counts:
        # Compute 3D color histograms for each query and support image
        histogram_list = get_3D_histogram_for_each_queryset(bin_count, True)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images.keys(), support_images.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images.keys(), support_images.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images.keys(), support_images.keys(), similarities_query3_support)

        # Print results for 3D color histograms with RGB values
        print(f"3D RGB Histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)}:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

# Calling function for 3D HSV Histograms
def pipeline_for_3D_HSV(bin_counts):
    # Perform experiments for 3D color histograms with HSV values
    for bin_count in bin_counts:
        # Compute 3D color histograms for each query and support image for hsv values
        histogram_list = get_3D_histogram_for_each_queryset(bin_count, False)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images_hsv.keys(), support_images_hsv.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images_hsv.keys(), support_images_hsv.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images_hsv.keys(), support_images_hsv.keys(), similarities_query3_support)

        # Print results for 3D color histograms with RGB values
        print(f"3D HSV Histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)}:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

# Calling function for per_channel RGB Histograms
def pipeline_for_per_channel_RGB(bin_counts):
    # Perform experiments for per_channel color histograms with RGB values
    for bin_count in bin_counts:
        # Compute per_channel color histograms for each query and support image
        histogram_list = get_per_channel_histogram_for_each_queryset(bin_count, True)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images.keys(), support_images.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images.keys(), support_images.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images.keys(), support_images.keys(), similarities_query3_support)

        # Print results for per_channel color histograms with RGB values
        print(f"Per channel RGB Histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)}:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

# Calling function for per_channel HSV Histograms
def pipeline_for_per_channel_HSV(bin_counts):
    # Perform experiments for per_channel color histograms with HSV values
    for bin_count in bin_counts:
        # Compute per_channel color histograms for each query and support image
        histogram_list = get_per_channel_histogram_for_each_queryset(bin_count, False)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images_hsv.keys(), support_images_hsv.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images_hsv.keys(), support_images_hsv.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images_hsv.keys(), support_images_hsv.keys(), similarities_query3_support)

        # Print results for per_channel color histograms with RGB values
        print(f"Per channel HSV Histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)}:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

def get_per_channel_histogram_of_one_image_with_grid(image, bin_count, grid_count):
    height = image.shape[0]
    width = image.shape[1]

    grid_histograms = []

    cell_height = height // grid_count
    cell_width = width // grid_count
    
    # Compute the histogram of the each grid cell of the image
    for i in range(grid_count):
        for j in range(grid_count):
            grid_cell = image[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width, 0:3]
            grid_histogram = get_per_channel_histogram(grid_cell, bin_count)
            grid_histograms.append(grid_histogram)
            
    return grid_histograms

def get_3D_histogram_of_one_image_with_grid(image, bin_count, grid_count):
    height = image.shape[0]
    width = image.shape[1]

    grid_histograms = []

    cell_height = height // grid_count
    cell_width = width // grid_count

    # Compute the histogram of the each grid cell of the image
    for i in range(grid_count):
        for j in range(grid_count):
            cell = image[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width, 0:3]
            grid_histograms.append(get_3D_histogram(cell, bin_count))

    return grid_histograms

def get_per_channel_histogram_grid_for_each_queryset(bin_count, grid_count):
    query1_histograms_with_grid = {}
    query2_histograms_with_grid = {}
    query3_histograms_with_grid = {}
    support_histograms_with_grid = {}
    # look for only hsv because it gave better results in previous experiments
    for keys in query1_images_hsv.keys():
        query1_histograms_with_grid[keys] = get_per_channel_histogram_of_one_image_with_grid(query1_images_hsv[keys], bin_count, grid_count)

    for keys in query2_images_hsv.keys():
        query2_histograms_with_grid[keys] = get_per_channel_histogram_of_one_image_with_grid(query2_images_hsv[keys], bin_count, grid_count)
    
    for keys in query3_images_hsv.keys():
        query3_histograms_with_grid[keys] = get_per_channel_histogram_of_one_image_with_grid(query3_images_hsv[keys], bin_count, grid_count)

    for keys in support_images_hsv.keys():
        support_histograms_with_grid[keys] = get_per_channel_histogram_of_one_image_with_grid(support_images_hsv[keys], bin_count, grid_count)
    
    return [query1_histograms_with_grid, query2_histograms_with_grid, query3_histograms_with_grid, support_histograms_with_grid]

def get_3D_histogram_grid_for_each_queryset(bin_count, grid_count):
    query1_histograms_with_grid = {}
    query2_histograms_with_grid = {}
    query3_histograms_with_grid = {}
    support_histograms_with_grid = {}
    # look for only hsv because it gives better results in previous experiments
    for keys in query1_images_hsv.keys():
        query1_histograms_with_grid[keys] = get_3D_histogram_of_one_image_with_grid(query1_images_hsv[keys], bin_count, grid_count)

    for keys in query2_images_hsv.keys():
        query2_histograms_with_grid[keys] = get_3D_histogram_of_one_image_with_grid(query2_images_hsv[keys], bin_count, grid_count)
    
    for keys in query3_images_hsv.keys():
        query3_histograms_with_grid[keys] = get_3D_histogram_of_one_image_with_grid(query3_images_hsv[keys], bin_count, grid_count)

    for keys in support_images_hsv.keys():
        support_histograms_with_grid[keys] = get_3D_histogram_of_one_image_with_grid(support_images_hsv[keys], bin_count, grid_count)
    
    return query1_histograms_with_grid, query2_histograms_with_grid, query3_histograms_with_grid, support_histograms_with_grid


# Calling function for per_channel grid histograms
# Since hsv gives better results at previous experiments, only hsv is used
def pipeline_for_per_channel_grid(grid_counts):
    bin_count = 32
    # Perform experiments for per_channel color histograms with HSV values
    for grid_count in grid_counts:
        # Compute per_channel color histograms for each query and support image
        histogram_list = get_per_channel_histogram_grid_for_each_queryset(bin_count, grid_count)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images_hsv.keys(), support_images_hsv.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images_hsv.keys(), support_images_hsv.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images_hsv.keys(), support_images_hsv.keys(), similarities_query3_support)

        # Print results for per_channel color histograms with RGB values
        print(f"Per channel grid histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)} and {grid_count}x{grid_count} grid size:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

# Calling function for 3D grid histograms
# Since hsv gives better results at previous experiments, only hsv is used
def pipeline_for_3D_grid(grid_counts):
    bin_count = 4
    # Perform experiments for 3D color histograms with HSV values
    for grid_count in grid_counts:
        # Compute per_channel color histograms for each query and support image
        histogram_list = get_3D_histogram_grid_for_each_queryset(bin_count, grid_count)
        # Compute image similarities for each query set
        similarities_query1_support = compute_image_similarity(histogram_list[0], histogram_list[3])
        similarities_query2_support = compute_image_similarity(histogram_list[1], histogram_list[3])
        similarities_query3_support = compute_image_similarity(histogram_list[2], histogram_list[3])

        # Calculate top-1 accuracy for each query set
        top1_accuracy_query1 = compute_top1_accuracy(query1_images_hsv.keys(), support_images_hsv.keys(), similarities_query1_support)
        top1_accuracy_query2 = compute_top1_accuracy(query2_images_hsv.keys(), support_images_hsv.keys(), similarities_query2_support)
        top1_accuracy_query3 = compute_top1_accuracy(query3_images_hsv.keys(), support_images_hsv.keys(), similarities_query3_support)

        # Print results for per_channel color histograms with RGB values
        print(f"3D Grid Histograms with bin count = {bin_count} / quantization interval = {int(256/bin_count)} and {grid_count}x{grid_count} grid size:")
        print(f"Top-1 Accuracy for Query 1: {top1_accuracy_query1 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 2: {top1_accuracy_query2 * 100:.2f}%")
        print(f"Top-1 Accuracy for Query 3: {top1_accuracy_query3 * 100:.2f}%")
        print("----------------------------------------")

# Experiments 
bin_counts_3D = [2, 4, 8, 16]
bin_counts_per_channel = [16, 32, 64, 128, 256]
pipeline_for_3D_RGB(bin_counts_3D)
pipeline_for_3D_HSV(bin_counts_3D)

pipeline_for_per_channel_RGB(bin_counts_per_channel)
pipeline_for_per_channel_HSV(bin_counts_per_channel)

grid_counts = [2, 4, 6, 8]
pipeline_for_3D_grid(grid_counts)
pipeline_for_per_channel_grid(grid_counts)






































