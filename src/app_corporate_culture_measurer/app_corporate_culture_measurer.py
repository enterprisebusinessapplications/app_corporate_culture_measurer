import logging
import sys
from logging import log
from pathlib import Path

from preprocess.parse import sequential_parse
from preprocess.clean import clean_file
from train.train import train
from core.create_dictionary import create_culture_dictionary
from core.score import score

import global_options

# from train.train import sequential_parse
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

if __name__ == "__main__":
    reparse = False
    reclean = False
    retrain = False
    regenerate_dictionary = False

    if reparse:
        sequential_parse()
    if reclean:
        # clean the parsed text (remove POS tags, stopwords, etc.) ----------------
        clean_file(
            in_file=Path(
                global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
            ),
            out_file=Path(
                global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
            ),
        )
    if retrain:
        train()
    if regenerate_dictionary:
        create_culture_dictionary()

    score()
# ...
