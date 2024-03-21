import datetime
import functools
import logging
import sys
from pathlib import Path

import pandas as pd

import global_options as global_options
import preprocess.parse as parse
from train import culture_models


def train():
    logging.log(logging.INFO, "training bigram model from corpus")
    culture_models.train_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
    )
    logging.log(logging.INFO, "completed training bigram model from corpus")
    culture_models.apply_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
        ),
        output_path=Path(
            global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
        scoring="original_scorer",
        threshold=global_options.PHRASE_THRESHOLD,
    )

    logging.log(logging.INFO, "training trigram model from bigrammed corpus")
    culture_models.train_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
    )
    culture_models.apply_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"
        ),
        output_path=Path(
            global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
        scoring="original_scorer",
        threshold=global_options.PHRASE_THRESHOLD,
    )
    logging.log(logging.INFO, "completed training trigram model from bigrammed corpus")


    logging.log(logging.INFO, "training trigram w2v model from trigrammed corpus")
    culture_models.train_w2v_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
        vector_size=global_options.W2V_DIM,
        window=global_options.W2V_WINDOW,
        workers=global_options.N_CORES,
        epochs=global_options.W2V_ITER,
    )
    logging.log(logging.INFO, "completed training trigram w2v model from trigrammed corpus")
