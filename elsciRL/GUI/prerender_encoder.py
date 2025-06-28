import torch
import numpy as np
from torch import Tensor
from elsciRL.encoders.language_transformers.MiniLM_L6v2 import LanguageEncoder as MiniLM_L6v2

# Get search method
import os
import json
from datetime import datetime


def encode_prerender_data(observed_states:dict|str=None,
                        save_dir:str=None,
                        encoder:str ='MiniLM_L6v2') -> Tensor:
    """    Encodes the observed states using a language encoder.
    Args:
        observed_states (dict or str): The observed states to encode, can be the dictionary or the directory path string.
        save_dir (str): The directory where the encoded states will be saved. If None, defaults to './encoded-prerender-data'.
        encoder (str): The name of the encoder to use. Defaults to 'MiniLM_L6v2', options include:
            - 'MiniLM_L6v2': A lightweight language model suitable for encoding text.
            - ~~Other encoders can be added in the future.~~
    Returns:
        Tensor: The encoded representation of the observed states.
    """
    # ------------------------------------------------------------------
    # Define the available encoders
    # Currently only MiniLM_L6v2 is available, but can be extended in the future.
    ENCODERS = {'MiniLM_L6v2': MiniLM_L6v2}
    encoder = ENCODERS[encoder]()
    # ------------------------------------------------------------------
    if observed_states is None:
        print("\n ----------------------------------------------------")
        print(" No observed states provided. Please select a file to encode.")
        print(" ----------------------------------------------------\n")
        file_names = [file for file in os.listdir('./') if file.endswith('.txt')]
        for n, file in enumerate(file_names):
            print(f"- {n}: {file}")
        selection = input("\n Select the file to encode (by number): ")
        observed_states_filename = file_names[int(selection)]
        observed_states_path = os.path.join('./', observed_states_filename)
        with open(observed_states_path, 'r') as f:
            observed_states = json.loads(f.read())
        save_dir = './'
    else:
        if isinstance(observed_states, str):
            observed_states_filename = observed_states.split('/')[-1].split('.')[0]
            if not save_dir:
                save_dir = os.path.dirname(observed_states)
            with open(observed_states, 'r') as f:
                observed_states = json.loads(f.read())
        else:
            observed_states_filename = 'observed_states'
            if not save_dir:
                save_dir = './'

    # Encode the observed states
    print(f"\n Encoding observed state file {observed_states_filename} using {encoder.name}...")
    str_states = [str_state for str_state in observed_states.values()]
    observed_states_encoded = encoder.encode(str_states)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'encoded_' + observed_states_filename.split('.')[0] + '.txt')
    np.savetxt(file_path, observed_states_encoded.numpy())
    print(f"Encoded states saved to {file_path}")
    print(f"Number of States: {len(observed_states_encoded)}")

    return observed_states_encoded