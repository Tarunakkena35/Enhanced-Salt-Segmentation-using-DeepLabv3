# Enhanced-Salt-Segmentation-using-DeepLabv3
## Abstract
Salt segmentation in seismic images is an essential process in the field of geophysics, particularly in the exploration and identification of subsurface resources like oil and gas. For companies looking for oil and gas, finding where the salt is matters a lot. Moreover, lands affected by salt become unsuitable for farming due to reduced plant absorption capacity, impacting growth rates. Seismic images are derived from sound waves bounced off underground structures, offer critical insights into the Earth's composition and the presence of valuable resources. However, these images may often contain salt formations. To identify salt-affected areas, seismic images are analysed at the pixel level to classify them as either salt or sediment. TGS Salt Identification Challenge dataset is used which consists of images captured from different underground spots randomly selected. Each image is 101 x 101 pixels and every pixel in these images is labelled as either salt or sediment. In this proposed model deeplabv3, utilizes ASPP, studies each tiny part of the picture, noticing textures and shapes to capture features at multiple scales. To figure out the model performance, we utilize metrics like Dice Similarity Coefficient, Intersection over Union (IOU).

## Workflow
1. Data Loading
2. Data Processing(Resizing the Image)
3. Data Splitting
4. Defining the Model DeepLabv3 with Resnet50
5. Training the Model
6. Evalution
7. Visualization

### Dataset Description
The dataset comprises a collection of images captured from diverse subsurface locations, selected randomly. These images are sized at 101 x 101 pixels, with each pixel categorized as either salt or sediment. Alongside the seismic images, the depth of each imaged location is included. The objective of the competition is to accurately delineate regions containing salt. The dataset comprises 4,000 images, divided into 3,200 images for training and 800 images for testing.

## Steps Involved
1. Importing pandas library as pd for data manipulation, then reading training and depth information CSV files into pandas DataFrames.
2. Merging depth and training dataframes, then creating a new column 'salt' to indicate presence of salt based on the 'rle_mask' column.
3. Importing zipfile and os modules for zip file handling and file operations respectively. Extracting contents of 'competition_data.zip' into '/kaggle/working/'.import zipfile
4. Importing glob for file path matching and os for operating system functionalities.
5. Importing necessary layers and modules from TensorFlow Keras for building neural networks and the Adam optimizer, and importing tqdm for progress monitoring.
6. Importing OpenCV as cv2 for computer vision tasks and numpy as np for numerical computations.
7. Importing necessary callbacks for monitoring and controlling the training process.
8. Importing Adam optimizer from TensorFlow and train_test_split function from scikit-learn for data splitting.
9. Importing numpy as np, TensorFlow as tf, and backend module from Keras for backend operations.
10. Assigning a small constant value for smoothing numerical operations.
11. Defining a function to calculate the Intersection over Union (IoU) score between true and predicted binary masks.
12. Defining a function to compute the Dice coefficient between true and predicted binary masks.
13. Defining a function to compute the Dice loss, which is 1 minus the Dice coefficient between true and predicted binary masks.
14. Importing necessary layers and modules from TensorFlow Keras for building neural network models.
15. Resizing train images in the specified folder and saving them to the output folder.
16. Path to the folder containing the images
17. Output folder for resized images
18. Declare the Target size
19. List all image files in the folder
20. Process each image
21. Resizing mask images in the specified folder and saving them to the output folder similarly.
22. Defining a function to implement the Atrous Spatial Pyramid Pooling (ASPP) module for semantic segmentation.
23. Defining a function to build the DeepLabV3 model architecture for semantic segmentation.
24. Defining a function to create a directory if it does not already exist.
25. Defining a function to load and split the dataset into training, testing, and validation sets.
26. Defining a function to read and preprocess an image.
27. Defining functions for reading and preprocessing images and masks, parsing them into TensorFlow tensors, and creating a TensorFlow dataset.
28. Import necessary libraries and define constants
29. Define the read_mask function
30. Define the tf_parse function
31. Define the tf_dataset function
32. Training the DeepLabV3 model on the dataset and evaluating its performance on the validation set.
33. Evaluating the trained model on the validation dataset and printing the evaluation results.
34. Saving the trained DeepLabV3 model to a file named 'deeplabv3_model.h5'.

