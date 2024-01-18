# Computer-Vision-Assignment-1
Instance Recognition with Color Histograms

This repo contains a Python file named vision_hw1.py for performing color histogram analysis in both RGB and HSV color spaces, using per-channel and 3D histograms, with an option for grid-based analysis. The experiments are designed to evaluate the effectiveness of different histogram methods in instance recognition task.

Prerequisites

Before running the experiments, ensure you have the following installed:
* Python 3.x
* NumPy
* Pillow (PIL Fork)

Dataset Preparation

1. Dataset Structure: Assumed dataset folder and Python file will be in same directory. The dataset should be structured as follows:
* dataset/: Dataset path 
* dataset/query_1/: Contains images for Query 1.
* dataset/query_2/: Contains images for Query 2.
* dataset/query_3/: Contains images for Query 3.
* dataset/support_96/: Contains support images.
* dataset/InstanceNames.txt: A text file listing the names of the images.


Running the Experiments

To run the experiments, follow these steps:

1. Open Terminal or Command Prompt: Navigate to the directory containing the script.

2. Execute the Script: Run the script using Python.
      python vision_hw1.py 
This will execute the predefined pipelines for both per-channel and 3D histogram analysis in RGB and HSV color spaces, with various bin and grid configurations.
* pipeline_for_3D_RGB(bin_counts_3D): Caller function for 3D RGB histograms
* pipeline_for_3D_HSV(bin_counts_3D): Caller function for 3D HSV histograms
* pipeline_for_per_channel_RGB(bin_counts_per_channel): Caller function for Per Channel RGB histograms
* pipeline_for_per_channel_HSV(bin_counts_per_channel): Caller function for Per Channel HSV histograms
* pipeline_for_3D_grid(grid_counts): Caller function for 3D histograms with grid based approach using best configurations 
* pipeline_for_per_channel_grid(grid_counts): Caller function for Per Channel histograms with grid based approach using best configurations 

3. View Results: The script will output the top-1 accuracy results for each query set in the terminal. 

Customization

You can customize the experiments by modifying the following parameters in the script:
* bin_counts_3D: List of bin counts for 3D histograms.
* bin_counts_per_channel: List of bin counts for per-channel histograms.
* grid_counts: List of grid sizes for grid-based analysis.
Adjust these parameters as needed to explore different configurations.


