import typing as T

import hashlib
import os
import pickle as pk
import sys
from pathlib import Path

import torch

import sys
sys.path.append("..")
from utils import get_logger
from .base import Featurizer

logg = get_logger()

MODEL_CACHE_DIR = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
)

FOLDSEEK_MISSING_IDX = 20

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class ESMFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=True, model_type='esm2_t6_8M_UR50D'):

        import esm

        if model_type == 'esm2_t6_8M_UR50D':
            (self._esm_model, self._esm_alphabet,) = esm.pretrained.esm2_t6_8M_UR50D()
            shape, self.repr_layers = 320, 6
        elif model_type == 'esm2_t12_35M_UR50D':
            (self._esm_model, self._esm_alphabet,) = esm.pretrained.esm2_t12_35M_UR50D()
            shape, self.repr_layers = 480, 12
        elif model_type == 'esm2_t30_150M_UR50D':
            (self._esm_model, self._esm_alphabet,) = esm.pretrained.esm2_t30_150M_UR50D()
            shape, self.repr_layers = 640, 30
        elif model_type == 'esm2_t33_650M_UR50D':
            (self._esm_model, self._esm_alphabet,) = esm.pretrained.esm2_t33_650M_UR50D()
            shape, self.repr_layers = 1280, 33
        else:
            raise Exception

        super().__init__("ESM", shape, save_dir)

        self._save_path = save_dir / Path(f"{self._name}_{model_type}_features.h5")

        torch.hub.set_dir(MODEL_CACHE_DIR)

        self._max_len = 1024
        self.per_tok = per_tok
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str):
        seq = seq.upper()
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(
            [("sequence", seq)]
        )
        batch_tokens = batch_tokens.to(self.device)
        results = self._cuda_registry["model"][0](
            batch_tokens, repr_layers=[self.repr_layers], return_contacts=True
        )
        token_representations = results["representations"][self.repr_layers]

        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        tokens = token_representations[0, 1 : len(seq) + 1]

        if self.per_tok:
            return tokens
        else: # Generate per-sequence representations via averaging
            return tokens.mean(0)