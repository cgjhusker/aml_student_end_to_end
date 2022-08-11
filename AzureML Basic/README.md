# AzureML Basic End to End

This end to end process will run through the steps of uploading, training, and deploying machine learning models to AzureML.

Prerequisites:
1) Have an AzureML account
2) Have a completed ml model using python
3) The python script uses the MLFlow Library

Steps:
1) Start by cloning the python files in this repository
2) Upload the python files into the notebooks section of AzureML
3) Create a Compute Instance in AzureML
4) Go back to the notebooks tab and run the train.py file
5) While this script runs a job within AzureML will be created due to MLFlow
6) Once this completes, open the models tab and click create new model, choose from a job output, and select the job that you just created
7) If you are not using a free account, skip to step 9
8) Before deploying your model to an endpoint, go and delete the compute instance you previously created
9) Now go to the endpoint tab and by selecting the model you just created, deploy this model to an endpoint
