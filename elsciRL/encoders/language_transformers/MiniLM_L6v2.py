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

    def encode(self, state: List[str], legal_actions:list = None, episode_action_history:list = None, 
               indexed: bool = False) -> Tensor:
        
        # I think typing is overriding the input type anyway -> need to ensure sentences are split up
        if type(state) == type(''):
            state = state.split(".") 
            state = [s for s in state if s.strip()]
        elif (len(state) == 0):
            state = [""]
                
        to_encode = [sent for sent in state if sent not in LanguageEncoder._cached_enc]
        if (to_encode):
            encoded = self.sentence_model.encode(to_encode, convert_to_tensor=True, show_progress_bar = False)
            LanguageEncoder._cached_enc.update({to_encode[i]: encoded[i] for i in range(len(to_encode))})
        
        LanguageEncoder._cached_freq.update(state)
        LanguageEncoder._cached_freq.subtract(LanguageEncoder._cached_freq.keys())
        state_encoded = torch.stack([LanguageEncoder._cached_enc[sent] for sent in state])

        if (len(LanguageEncoder._cached_freq) > 10000):
            for key, freq in list(reversed(LanguageEncoder._cached_freq.most_common()))[:2000]:
                del LanguageEncoder._cached_enc[key]
                del LanguageEncoder._cached_freq[key]

        return state_encoded