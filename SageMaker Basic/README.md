# SageMaker Basic End to End Scenario

This repo will help you upload, train, and deploy a machine learning model to AWS SageMaker

# Prerequisites
1) Have an AWS account

# Steps
1) Download the tar.gz file within the dist folder, or create a new one by packaging the task.py file
2) Upload the tar.gz file to AWS s3 bucket
3) Create a PyTorch stack here: https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=torchserve-on-aws&templateURL=https://torchserve-workshop.s3.amazonaws.com/torchserve-workshop-template.yaml
4) Open AWS SageMaker and click the notebooks tab
5) Open the newly created torchserve notebook
6) Update the s3 path to the tar.gz file first code block
7) Update the python file in the second code block
8) Run the first 3 code blocks
9) All done! 
