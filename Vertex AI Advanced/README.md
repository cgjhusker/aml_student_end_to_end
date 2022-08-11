# Vertex AI Advanced End to End Scenario

The code in this repo can be used to create and run machine learning components and pipelines within Vertex AI

## Prerequisites
1) Have a Vertex AI account

## Steps:
1) Download the notebook file in this repository
2) Update the files bucket name, uri, service account, and project id
3) Edit the @component code blocks with the desired python code for the component
4) Run the entire notebook
5) Go back to Vertex AI and see the new pipeline

## Description

This notebook defines two components and organizes them into a pipeline. Vertex AI defines a component using a normal python function along with an @commponent tag.
Then to define the workflow of the pipeline, simply create a new python function calling the component functions as in normal python. Now after running the compile and run commands the pipeline will be created.
