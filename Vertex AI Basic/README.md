# Vertex AI Basic End to End Scenario

# Prerequisites
1) Docker Installed
2) A Vertex AI account

# Steps
1) Download the files in this repository
2) Package the python files into the tar.gz or use the one provided in this repo
3) Download your machine learning file, I have a pytorch model saved here as a .pt file
4) Create a custom handler that defines how the model will be interacted with once deployed (process data, export data, etc)
5) Upload this custom handler to docker and create a torchserve container
6) Push this container to Vertex AI
7) Upload the tar.gz file to GCP Cloud Storage within a custom bucket
8) Add an output folder to this bucket
9) Then create a model folder within that output folder
10) Upload the saved model file to this model folder (ie the .pt file)
11) Now go into Vertex AI and create a new training job, specify the tar.gz file, the output file, and the built in container for training
12) Then when Vertex AI asks for a prediction container select the custom image you pushed from docker
13) Now wait for the model to finish

# Additional Notes
For this walkthrough all the steps consisting of a docker image, custom handler, torchserve, and prediction containers can be avoided if the user selects one of the supported python machine learning frameworks (skleanr, xgboost, or tensorflow). I would highly recommend using this option as it greatly reduces the steps to complete. Then all you would need to do is create the tar.gz file.
