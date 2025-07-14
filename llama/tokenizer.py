# tokenizer.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from typing import List
from sentencepiece import SentencePieceProcessor

# If you want optional logging on Windows, you can use print or Python's logging module.
import logging
logger = logging.getLogger(__name__)

class Tokenizer:
    """
    A simple wrapper around a SentencePiece tokenizer for encoding/decoding text.

    Usage:
        tokenizer = Tokenizer("path/to/tokenizer.model")
        tokens = tokenizer.encode("Hello world!", bos=True, eos=True)
        text = tokenizer.decode(tokens)
    """

    def __init__(self, model_path: str):
        """
        Initialize the Tokenizer using a SentencePiece model file.

        Args:
            model_path (str): Path to the SentencePiece model (.model file).
        """
        # Check that the file exists before loading
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        # Load the SentencePiece model
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Loaded SentencePiece model from {model_path}")

        # Store vocabulary and special token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        logger.info(
            f"Tokenizer vocab size: {self.n_words} | BOS ID: {self.bos_id} | EOS ID: {self.eos_id} | PAD ID: {self.pad_id}"
        )
        # Double-check that the reported vocab size matches the model
        assert self.n_words == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            s (str): Text to encode.
            bos (bool): If True, prepend BOS token.
            eos (bool): If True, append EOS token.

        Returns:
            List[int]: List of token IDs.
        """
        if not isinstance(s, str):
            raise TypeError("Input to encode() must be a string.")
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            t (List[int]): List of token IDs.

        Returns:
            str: Decoded string.
        """
        if not isinstance(t, list):
            raise TypeError("Input to decode() must be a list of ints.")
        return self.sp_model.decode(t)

# Usage tip:
# - Make sure you have sentencepiece installed: pip install sentencepiece
# - You can train your own tokenizer.model with the sentencepiece command line or use a prebuilt one.
