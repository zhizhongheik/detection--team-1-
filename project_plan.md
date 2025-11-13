# Task List and Deadlines

Phase 1 (October 30th to November 6th): Develop a plan

Phase 2 (November 6th to November 13th): Extract keyframes from videos, select images to create a dataset, and manually annotate the images (output YOLO format)

Phase 3 (November 13th to November 20th): Complete model training and test whether the model achieves the expected output results

# Role Assignment

Phase 1:

A: Create a GitHub collaboration platform and join B

B: Prepare scripts, write a project plan, README file, etc.

Phase 2:

A: Extract at least 54 images from keyframes in two dust videos, and select 50 usable photos as the dataset

B: Use Roboflow to manually annotate each image, generating a labeled YOLO dataset

Phase 3:

A: Train and validate the model

B: Test the model and refine the plan

# Tool Selection and Reasons

Roboflow: Simple and easy to use, supports collaborative creation and output of YOLO files

VScode: Convenient for connecting to GitHub and creating a collaborative cloud platform

# Expected Outputs at Each Stage

Image Labeling: Use Roboflow to find at least 100 images, and select at least 50 suitable for training and validation. Requirements: (Dust must be clearly visible in the image, and there should be no duplicate images).

Model Training: Train the model using YOLOv8, aiming for an accuracy of over 80%. Test the images to see if the expected output is achieved.

Inference Results: Summarize and improve the problems encountered during model training. The model should be able to accurately identify and locate dust in test videos with high accuracy.

Database: Docker
