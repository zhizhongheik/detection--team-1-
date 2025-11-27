# Task List and Deadlines

Phase 1 (October 30, 2025 - November 6, 2025): Plan Development

Phase 2 (November 6, 2025 - November 13, 2025): Extract keyframes from videos, create a dataset by selecting images, and manually annotate the images (output in YOLO format).

Phase 3 (November 13, 2025 - November 20, 2025): Complete model training and test whether the model achieves the expected output results.

Phase 4 (November 20, 2025 - November 27, 2025): Complete the backend API setup and Docker data containerization deployment.

Phase 5 (November 27, 2025 - December 4, 2026): Complete the PowerPoint presentation and report.

# Role Assignment

Phase 1:

A: Create a GitHub collaboration platform and join B's team, prepare scripts, README files, etc., and collaborate with B to write reports and create PPTs. Phase Two:

A: Extract at least 54 images from keyframes of two dust videos and select 50 usable images as the dataset.

B: Manually annotate each image using Roboflow to generate a labeled YOLO dataset.

Phase Three:

A: Train and validate the model.

B: Test the model; model building and optimization.

Phase Four:

A: Set up the FastAPI backend and test the model accuracy.

B: Deploy Docker data containerization.

Phase Five:

Collaborate on report writing and PPT creation.

# Tool Selection and Reasons

Using Roboflow for image annotation: Simple and easy to use, supports collaborative creation and output of YOLO files;

VS Code: Convenient for connecting to GitHub and creating collaborative cloud platforms;

MySQL: Free and easy-to-use database with simple operation.


# Expected Output at Each Stage

Image Annotation: Use Roboflow to find at least 100 images and select at least 50 for training and validation. Requirements: Dust must be clearly visible in the images, and images must not be duplicated.

Model Training: Train the model using YOLOv8m, aiming for an accuracy exceeding 80%. Test images to verify if the expected output has been achieved.

Results: The model achieved an accuracy exceeding 85%, accurately identifying aluminum powder in the final test. Summarize and improve the problems encountered during model training. The model should be able to accurately identify and locate dust in the test video.

Backend Development: A web interface was built using FastAPI, and a database was built using MySQL, enabling users to successfully upload images or videos and save them to a specified path. In the final test webpage, after a user uploads an image or video, the server responds and outputs the result.

Final Presentation: Compile a final report covering the operation process, training accuracy, display of training results, and improvement suggestions, and finally create a presentation.

Database: Docker, MySQL
