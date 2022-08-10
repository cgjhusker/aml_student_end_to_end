import os
import json
import logging

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class MNISTFashionClassifier(BaseHandler):
    """
    The handler takes an input string and returns the classification text 
    based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(MNISTFashionClassifier, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ Loads the model.pt file and initializes the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "model.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "neural_network.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        from trainer.neural_network import NeuralNetwork
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = NeuralNetwork()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """

        return data

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        prediction = self.model(inputs)

        logger.info("Model predicted: '%s'", prediction)
        return [prediction]

    def postprocess(self, inference_output):
        return inference_output