import argparse
import numpy as np
import sys
import time
import tritonclient.grpc as grpcclient
import os

class BaseTriton():
    def __init__(self, input_name=[], input_type=[], input_dim=[], output_name=[], \
                url='', verbose=False, ssl=False, root_certificates=None, private_key=None, \
                certificate_chain=None, client_timeout=None, grpc_compression_algorithm=None, static=False):
        self.triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)

        self.input_name = input_name # 'data'
        self.input_type = input_type # "FP32"
        self.input_dim = input_dim # [1, 3, 112, 112]
        self.output_name = output_name
        self.client_timeout = client_timeout
        self.static = static

    def predict(self, inputs_data, model_name):
        # Infer
        inputs = []
        outputs = []
        for i, input_data in enumerate(inputs_data):
            inputs.append(grpcclient.InferInput(self.input_name[i], input_data.shape, self.input_type[i]))

            # Initialize the data
            inputs[i].set_data_from_numpy(input_data)

        for i, out_name in enumerate(self.output_name):
            outputs.append(grpcclient.InferRequestedOutput(out_name))

        # Test with outputs
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.client_timeout,
            headers={'test': '1'})
        
        if self.static:
            statistics = self.triton_client.get_inference_statistics(model_name=model_name)
            print(statistics)
            if len(statistics.model_stats) != 1:
                print("FAILED: Inference Statistics")
                sys.exit(1)

        # Get the output arrays from the results
        outputs_data = [results.as_numpy(self.output_name[i]) for i in range(len(self.output_name))]
        return outputs_data


