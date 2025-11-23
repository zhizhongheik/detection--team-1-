# Task List and Deadlines

Phase 1 (October 30th - November 6th, 2025): Plan Development

Phase 2 (November 6th - November 13th, 2025): Extract keyframes from videos, create a dataset by selecting images, and manually label the images (output in YOLO format).

Phase 3 (November 13th - November 20th, 2025): Complete model training and test whether the model achieves the expected output results.

Phase 4 (November 21st, 2025 - December 4th, 2026): Complete the PowerPoint presentation and report.

# Role Assignment

Phase 1:

A: Create a GitHub collaboration platform and join B, prepare scripts, README files, etc., and work with B to write a report and create a PPT.

B: Prepare scripts, write a project plan, README files, etc., and work with A to write a report and create a PPT.

Phase 2:

A: Extract at least 54 images from keyframes of two dust videos and select 50 usable photos as the dataset.

B: Manually annotate each image using Roboflow to generate a labeled YOLO dataset.

Phase 3:

A: Train and validate the model.

B: Test the model; model building and scheme optimization.

Phase 4:

Write the report and create a PPT.

# Tool Selection and Reasons

Use Roboflow for image annotation: Simple and easy to use, supports collaborative creation and output of YOLO files;

VS Code: Convenient for connecting to GitHub and creating a collaborative cloud platform;

MySQL: Free, easy-to-use database, easy to operate.

# Expected Output at Each Stage

Image Annotation: Use Roboflow to find at least 100 images and select at least 50 for training and validation. Requirements: Dust must be clearly visible in the images, and images must not be duplicated.

Model Training: Train the model using YOLOv8m, aiming for an accuracy exceeding 80%. Test images to verify if the expected output is achieved.

Results: The model achieved an accuracy of over 85%, accurately identifying aluminum dust in the final test. Summarize and improve upon issues encountered during model training. The model should be able to accurately identify and locate dust in the test video.

Database: Docker, MySQL
