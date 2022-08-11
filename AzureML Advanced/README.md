# AzureML Advanced End to End Scenario

In this repository is the code to set up, create, and run components and pipelines within AzureML. This allows a user to create reproducible workflow components to better manage their Machine Learning Lifecycle.

Prerequisites:
1) Have the Azure CLI installed
2) Be authenticated within the Azure CLI
3) Have either a free or subscription account with Azure
4) Have some compute target setup on Azure

Steps:
1) Clone the contents of this repo
2) Run two cli commands to upload the commponents to AzureML

          a) az ml component create -f model/model.yaml 
  
          b) az ml component create -f create/create.yaml 
  
3) Open up AzureML and go to Designer
4) Drag into the Canvas the Compoents you wish to use and connect them accordingly
5) Go to the setting panel in the top right and set a compute target
6) Now you are all set to run your Pipeline!
