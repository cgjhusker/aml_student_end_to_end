# SageMaker Advanced End to End Scenario

In this end to end process, these steps will demonstrate how to create machine learning components and pipelines within SageMaker.

## Prerequisites:
1) Have an AWS account

## Steps:
1) Download the notebook file in this repository
2) Upload the notebook to AWS
3) Run the notebook

These are all the steps you need to create a pipeline, but here's some more information about what is going on in this notebook.

## Extra Knowledge:

What this notebook is doing is creating two python files (preprocess and evaluate) to act as the components. 
Then the file creates two processing objects to control the environments of the components, specify the correct python file, inputs/outputs, and declare any dependencies like a machine learning framework. 
Finally the notebook defines a pipeline to be the components and how to connect them, then simply calling a run command creates the pipeline.
