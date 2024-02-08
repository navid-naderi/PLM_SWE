import typing as T
from types import SimpleNamespace

import os
import pickle as pk
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.random import choice
from sklearn.model_selection import KFold, train_test_split
from tdc.benchmark_group import dti_dg_group
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from featurizer import Featurizer
from featurizer.protein import FOLDSEEK_MISSING_IDX
from utils import get_logger

from Bio import SeqIO
import requests as r
from Bio import SeqIO
from io import StringIO

logg = get_logger()

def retrieve_protein_seq(ID):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+ID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq,'fasta'))

    return str(pSeq[0].seq)


def get_task_dir(task_name: str, database_root: Path):
    """
    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    database_root = Path(database_root).resolve()

    task_paths = {
        "bindingdb": database_root / "BindingDB",
        "davis": database_root / "DAVIS",
        "dti_dg": database_root / "TDC",
        "ppi_gold": database_root / "PPI_Gold",
    }

    return Path(task_paths[task_name.lower()]).resolve()


def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    d_emb = [a[0] for a in args]
    t_emb = [a[1] for a in args]
    labs = [a[2] for a in args]

    drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
    labels = torch.stack(labs, 0)
    return drugs, targets, labels

def ppi_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader

    If protein embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    p0_emb = [a[0] for a in args]
    p1_emb = [a[1] for a in args]
    labs = [a[2] for a in args]
    proteins = pad_sequence(p0_emb + p1_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX) # first half of the batch are the first sequences in the PPI pair
    labels = torch.stack(labs, 0)
    return proteins, labels

class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i])
        return drug, target, label

class BinaryPPIDataset(Dataset):
    def __init__(
        self,
        targets_pos_0,
        targets_pos_1,
        targets_neg_0,
        targets_neg_1,
        target_featurizer: Featurizer,
    ):
        self.proteins_0 = targets_pos_0 + targets_neg_0
        self.proteins_1 = targets_pos_1 + targets_neg_1
        self.labels = torch.cat((torch.ones(len(targets_pos_0),), torch.zeros(len(targets_pos_0),)), dim=0)
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        p0 = self.target_featurizer(self.proteins_0[i])
        p1 = self.target_featurizer(self.proteins_1[i])
        label = torch.tensor(self.labels[i])
        return p0, p1, label

class DTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")

        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):
        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )

        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class PPIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=" ",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": ppi_collate_fn,
        }

        self._csv_kwargs = {
            "header": None,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path_pos = Path("Intra1_pos_rr.txt")
        self._train_path_neg = Path("Intra1_neg_rr.txt")
        self._val_path_pos = Path("Intra0_pos_rr.txt")
        self._val_path_neg = Path("Intra0_neg_rr.txt")
        self._test_path_pos = Path("Intra2_pos_rr.txt")
        self._test_path_neg = Path("Intra2_neg_rr.txt")

        self._target_columns = [0, 1]
        self._label_column = "Label"

        self.target_featurizer = target_featurizer

    def convert_pID_to_seq(self, ID_list):

        fasta_file = self._data_dir / Path('human_swissprot_oneliner.fasta')
        fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
        fasta_dict = {}
        for fasta in fasta_sequences:
            name, seq = fasta.id, str(fasta.seq)
            fasta_dict[name] = seq

        seq_list = []
        for pID in ID_list:
            try:
                seq_list.append(fasta_dict[pID])
            except:
                seq = retrieve_protein_seq(pID)
                fasta_dict[pID] = seq # add retrieved sequence to the fasta dictionary
                seq_list.append(seq)

        return seq_list

    def get_all_target_seqs(self):

        self.df_train_pos = pd.read_csv(self._data_dir / self._train_path_pos, **self._csv_kwargs)
        self.df_train_neg = pd.read_csv(self._data_dir / self._train_path_neg, **self._csv_kwargs)

        self.df_val_pos = pd.read_csv(self._data_dir / self._val_path_pos, **self._csv_kwargs)
        self.df_val_neg = pd.read_csv(self._data_dir / self._val_path_neg, **self._csv_kwargs)

        self.df_test_pos = pd.read_csv(self._data_dir / self._test_path_pos, **self._csv_kwargs)
        self.df_test_neg = pd.read_csv(self._data_dir / self._test_path_neg, **self._csv_kwargs)

        self.dataframes = [self.df_train_pos, self.df_train_neg, self.df_val_pos, self.df_val_neg, self.df_test_pos, self.df_test_neg]

        all_targets_ID = pd.concat([i[col] for i in self.dataframes for col in self._target_columns]).unique()
        self.all_targets = self.convert_pID_to_seq(all_targets_ID)

    def prepare_data(self):
        if self.target_featurizer.path.exists():
            logg.warning("Target featurizers already exist")
            return

        self.get_all_target_seqs()
                
        if self._device.type == "cuda":
            self.target_featurizer.cuda(self._device)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(self.all_targets)

        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):

        self.get_all_target_seqs()
        
        if self._device.type == "cuda":
            self.target_featurizer.cuda(self._device)

        self.target_featurizer.preload(self.all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryPPIDataset(
                self.convert_pID_to_seq(self.df_train_pos[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_train_pos[self._target_columns[1]]),
                self.convert_pID_to_seq(self.df_train_neg[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_train_neg[self._target_columns[1]]),
                self.target_featurizer,
            )

            self.data_val = BinaryPPIDataset(
                self.convert_pID_to_seq(self.df_val_pos[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_val_pos[self._target_columns[1]]),
                self.convert_pID_to_seq(self.df_val_neg[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_val_neg[self._target_columns[1]]),
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryPPIDataset(
                self.convert_pID_to_seq(self.df_test_pos[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_test_pos[self._target_columns[1]]),
                self.convert_pID_to_seq(self.df_test_neg[self._target_columns[0]]),
                self.convert_pID_to_seq(self.df_test_neg[self._target_columns[1]]),
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class TDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._seed = seed

        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")

        train_val, test = (
            dg_benchmark["train_val"],
            dg_benchmark["test"],
        )

        all_drugs = pd.concat([train_val, test])[self._drug_column].unique()
        all_targets = pd.concat([train_val, test])[self._target_column].unique()

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):
        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")
        dg_name = dg_benchmark["name"]

        self.df_train, self.df_val = dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="default", seed=self._seed
        )
        self.df_test = dg_benchmark["test"]

        self._dataframes = [self.df_train, self.df_val]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)
