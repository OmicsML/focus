import json
import logging
import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab
from typing_extensions import Self

from .. import main_info
from .._settings import settings


class GeneVocab(Vocab):
    """
    Vocabulary for genes.

    This code is based on or includes code from: https://github.com/bowang-lab/scGPT
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = PAD_TOKEN,
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a gene list.

        Args:
            gene_list_or_vocab (List[str] or Vocab): List of gene names or a
                Vocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to PAD_TOKEN,
                if PAD_TOKEN is in the vocabulary.
        """
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError("receive non-empty specials when init from a Vocab object.")
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError("gene_list_or_vocab must be a list of gene names or a Vocab object.")
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                if isinstance(vocab, dict):
                    return cls.from_dict(vocab)
                elif isinstance(vocab, list):
                    return cls(vocab)
                else:
                    raise ValueError(f"{file_path} is not a valid file type.")
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(f"{file_path} is not a valid file type. " "Only .pkl and .json are supported.")

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = PAD_TOKEN,
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        Build a Vocab from an iterator. This function is modified from
        torchtext.vocab.build_vocab_from_iterator. The original function always
        splits tokens into characters, which is not what we want.

        Args:
            iterator (Iterable): Iterator used to build Vocab. Must yield list
                or iterator of tokens.
            min_freq (int): The minimum frequency needed to include a token in
                the vocabulary.
            specials (List[str]): Special symbols to add. The order of supplied
                tokens will be preserved.
            special_first (bool): Whether to add special tokens to the beginning

        Returns:
            torchtext.vocab.Vocab: A `Vocab` object
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])


def _build_default_gene_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> GeneVocab:
    """
    Build the default gene vocabulary from HGNC gene symbols.

    Args:
        download_source_to (str): Directory to download the source data.
        save_vocab_to (Path or str): Path to save the vocabulary. If None,
            the vocabulary will not be saved. Default to None.
    """
    gene_collection_file = Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
    if not gene_collection_file.exists():
        # download and save file from url
        url = (
            "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
            "col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
            "=on&order_by=gd_app_sym_sort&format=text&submit=submit"
        )
        import requests

        r = requests.get(url)
        gene_collection_file.write_text(r.text)

    main_info(f"Building gene vocabulary from {gene_collection_file}")
    df = pd.read_csv(gene_collection_file, sep="\t")
    gene_list = df["Approved symbol"].dropna().unique().tolist()
    gene_vocab = GeneVocab(gene_list)  # no special tokens set in default vocab
    if save_vocab_to is not None:
        gene_vocab.save_json(Path(save_vocab_to))
    return gene_vocab


def map_genes(
    adata: anndata.AnnData,
    vocab_file: Optional[Union[Path, str]] = None,
    vocab_dict: Optional[Dict[str, int]] = None,
    gene_col: str = "gene_name",
) -> Tuple[Union[GeneVocab, Vocab], anndata.AnnData]:
    """
    Map gene names to the vocabulary of the model.
    This is done by matching the gene names to the vocabulary of the model,
    and adding the special tokens to the vocabulary if they are not already present.

    Args:
        adata: The AnnData object to be processed.
        vocab_file: The path to the vocabulary file.
        gene_col: The column name of the gene names in adata.var.

    Returns:
        adata: The AnnData object has been processed.
    """

    if vocab_dict is not None:
        vocab = Vocab(vocab_dict)
    else:
        vocab = GeneVocab.from_file(vocab_file)
        if MASK_TOKEN not in vocab:
            vocab.append_token(MASK_TOKEN)
        if CLS_TOKEN not in vocab:
            vocab.append_token(CLS_TOKEN)
        if START_TOKEN not in vocab:
            vocab.append_token(START_TOKEN)
        vocab.set_default_index(vocab[MASK_TOKEN])

    if adata is not None:
        adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        adata.var[settings.gene_name_key] = adata.var[gene_col]
        main_info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

    return vocab, adata


def align_gene_dicts(
    vocab_1: dict,
    vocab_2: dict,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Map gene names from one gene dict to another. (vocab_1 -> vocab_2)

    Args:
        vocab_1: The gene dict of the reference dataset.
        vocab_2: The gene dict of the target dataset.
        max_length: The maximum length of the returned array. If None, will be set to
            the max value in vocab_1 + 1.

    Returns:
        align_list: A numpy array of gene ids (index) in vocab 1 to gene ids (value) in vocab 2.
    """

    if max_length is None:
        max_length = max(list(vocab_1.values())) + 1
    align_list = [-1] * max_length
    vocab_1_list = vocab_1.keys()
    vocab_2_list = vocab_2.keys()
    for token in vocab_1_list:
        if token in vocab_2_list:
            align_list[vocab_1[token]] = vocab_2[token]
        else:
            align_list[vocab_1[token]] = -1

    return np.array(align_list)