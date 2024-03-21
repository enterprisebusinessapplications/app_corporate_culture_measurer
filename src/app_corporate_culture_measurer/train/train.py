import logging
from pathlib import Path
import global_options as global_options
from train import culture_models


def train():

    def train_ngram_model(input_path, model_path):
        culture_models.train_ngram_model(input_path, model_path)

    def apply_ngram_model(input_path, output_path, model_path, threshold, scoring):
        culture_models.apply_ngram_model(
            input_path, output_path, model_path, threshold, scoring
        )

    logging.log(logging.INFO, "training bigram model from corpus")
    train_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
    )
    logging.log(logging.INFO, "completed training bigram model from corpus")

    logging.log(logging.INFO, "applying bigram model to corpus to produce a bigrammed corpus")
    apply_ngram_model(
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
    logging.log(
        logging.INFO, "completed applying bigram model to corpus to produce a bigrammed corpus"
    )

    logging.log(logging.INFO, "training trigram model from bigrammed corpus")
    train_ngram_model(
        input_path=Path(
            global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"
        ),
        model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
    )
    logging.log(logging.INFO, "completed training trigram model from bigrammed corpus")

    logging.log(
        logging.INFO, "applying trigram model to corpus to produce a trigrammed corpus"
    )
    apply_ngram_model(
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
    logging.log(
        logging.INFO,
        "completed applying trigram model to corpus to produce a trigrammed corpus",
    )

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
    logging.log(
        logging.INFO, "completed training trigram w2v model from trigrammed corpus"
    )
