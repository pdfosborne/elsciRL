import numpy as np
import torch
from multiprocessing.spawn import import_main_path
from typing import Dict, List, Tuple
from collections import Counter
from gymnasium.spaces import Box

from torch import Tensor
from elsciRL.encoders.encoder_abstract import StateEncoder

# Language Encoder
from sentence_transformers import SentenceTransformer



class LanguageEncoder(StateEncoder):
    """Required Language Model included in requisite packages."""
    _cached_enc: Dict[str, Tensor] = dict()
    _cached_freq: Counter = Counter()

    def __init__(self, device: str = None):
        autodev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device if device else autodev
        self.sentence_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        low_array = [-1 for i in range(384)]
        high_array = [1 for i in range(384)]
        self.observation_space = Box(low=np.array(low_array), high=np.array(high_array), dtype=np.float32)
        self.name = "MiniLM_L6v2"
        self.input_type = "text"
        self.output_type = "tensor"
        self.output_dim = 384

    def encode(self, state: str|List[str], legal_actions:list = None, episode_action_history:list = None, 
               indexed: bool = False, progress_bar:bool=False) -> Tensor:
        
        # I think typing is overriding the input type anyway -> need to ensure sentences are split up
        if type(state) == str:
            state = [state]
        #     state = state.split(".") 
        #     state = [s for s in state if s.strip()]
        if (len(state) == 0):
            state = [""]
        to_encode = [sent for sent in state if sent not in LanguageEncoder._cached_enc]
        if (to_encode):
            # Show progress bar if state is a list of strings
            encoded = self.sentence_model.encode(to_encode, batch_size=256, convert_to_tensor=True, show_progress_bar=progress_bar)
            LanguageEncoder._cached_enc.update({to_encode[i]: encoded[i] for i in range(len(to_encode))})
        
        LanguageEncoder._cached_freq.update(state)
        LanguageEncoder._cached_freq.subtract(LanguageEncoder._cached_freq.keys())
        state_encoded = torch.stack([LanguageEncoder._cached_enc[sent] for sent in state])

        if (len(LanguageEncoder._cached_freq) > 10000):
            for key, freq in list(reversed(LanguageEncoder._cached_freq.most_common()))[:2000]:
                del LanguageEncoder._cached_enc[key]
                del LanguageEncoder._cached_freq[key]

        return state_encoded