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
    # train and apply a phrase model to detect 2-word phrases ----------------
    
    culture_models.train_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
    )
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

    # train and apply a phrase model to detect 3-word phrases ----------------
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

    # train the word2vec model ----------------
    print(datetime.datetime.now())
    print("Training w2v model...")
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
