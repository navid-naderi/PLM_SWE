from __future__ import annotations

import typing as T

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append("..")
from utils import get_logger

logg = get_logger()

import multiprocessing
import time
import h5py_cache as h5c
np.float = float

def sanitize_string(s):
    return s.replace("/", "|")


def get_featurizer(featurizer_string, *args, **kwargs):
    import featurizer as featurizers

    featurizer_string_list = featurizer_string.split(",")
    if len(featurizer_string_list) > 1:
        featurizer_list = [
            getattr(featurizers, i.strip()) for i in featurizer_string_list
        ]
        return featurizers.ConcatFeaturizer(featurizer_list, *args, **kwargs)
    else:
        return getattr(featurizers, featurizer_string_list[0])(*args, **kwargs)


###################
# Base Featurizer #
###################


class Featurizer:
    def __init__(self, name: str, shape: int, save_dir: Path = Path().absolute()):
        self._name = name
        self._shape = shape
        self._save_dir = save_dir
        self._save_path = save_dir / Path(f"{self._name}_features.h5")

        self._preloaded = False
        self._device = torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}

    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)

        return self._features[seq]

    def _register_cuda(self, k: str, v, f=None):
        """
        Register an object as capable of being moved to a CUDA device
        """
        self._cuda_registry[k] = (v, f)

    def _transform(self, seq: str) -> torch.Tensor:
        raise NotImplementedError

    def _update_device(self, device: torch.device):
        self._device = device
        for k, (v, f) in self._cuda_registry.items():
            if f is None:
                try:
                    self._cuda_registry[k] = (v.to(self._device), None)
                except RuntimeError as e:
                    logg.error(e)
                    logg.debug(device)
                    logg.debug(type(self._device))
                    logg.debug(self._device)
            else:
                self._cuda_registry[k] = (f(v, self._device), f)
        for k, v in self._features.items():
            self._features[k] = v.to(device)

    @lru_cache(maxsize=5000)
    def transform(self, seq: str) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            feats = self._transform(seq)
            if self._on_cuda:
                feats = feats.to(self.device)
            return feats

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def path(self) -> Path:
        return self._save_path

    @property
    def features(self) -> dict:
        return self._features

    @property
    def on_cuda(self) -> bool:
        return self._on_cuda

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> Featurizer:
        self._update_device(device)
        self._on_cuda = device.type == "cuda"
        return self

    def cuda(self, device: torch.device) -> Featurizer:
        """
        Perform model computations on CUDA, move saved embeddings to CUDA device
        """
        self._update_device(device)
        self._on_cuda = True
        return self

    def cpu(self) -> Featurizer:
        """
        Perform model computations on CPU, move saved embeddings to CPU
        """
        self._update_device(torch.device("cpu"))
        self._on_cuda = False
        return self

    def write_to_disk(self, seq_list: T.List[str], verbose: bool = True) -> None:
        logg.info(f"Writing {self.name} features to {self.path}")
        with h5py.File(self._save_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                seq_h5 = sanitize_string(seq)
                if seq_h5 in h5fi:
                    logg.warning(f"{seq} already in h5file")
                feats = self.transform(seq)
                dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                dset[:] = feats.cpu().numpy()

    def load(self, load_inputs):
        save_path, seq = load_inputs

        h5fi = h5py.File(save_path, "r", driver='core')
        seq_h5 = sanitize_string(seq)
        if seq_h5 in h5fi:
            feats = torch.from_numpy(h5fi[seq_h5][:])
        else:
            feats = self.transform(seq)

        if self._on_cuda:
            feats = feats.to(self.device)

        return feats

    def preload(
        self,
        seq_list: T.List[str],
        verbose: bool = True,
        write_first: bool = True,
    ) -> None:
        logg.info(f"Preloading {self.name} features from {self.path}")

        if 'Morgan' in self.name:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                feats = self.transform(seq)
                if self._on_cuda:
                    feats = feats.to(self.device)
                self._features[seq] = feats

        else:
            if write_first and not self._save_path.exists():
                self.write_to_disk(seq_list, verbose=verbose)

            if self._save_path.exists():
                h5fi = h5py.File(self._save_path, "r", driver='core')

                for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                    seq_h5 = sanitize_string(seq)
                    try:
                        feats = torch.from_numpy(h5fi[seq_h5][:])
                    except:
                        feats = self.transform(seq)

                    if self._on_cuda:
                        feats = feats.to(self.device)

                    self._features[seq] = feats

            else:
                for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                    feats = self.transform(seq)

                    if self._on_cuda:
                        feats = feats.to(self.device)

                    self._features[seq] = feats

        self._update_device(self.device)
        self._preloaded = True
