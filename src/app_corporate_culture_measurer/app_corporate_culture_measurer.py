import logging
import sys
from logging import log
from pathlib import Path

from preprocess.parse import sequential_parse
from preprocess.clean import clean_file
import global_options

# from train.train import sequential_parse

if __name__ == "__main__":
    sequential_parse()
    # clean the parsed text (remove POS tags, stopwords, etc.) ----------------
    clean_file(
        in_file=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
        ),
        out_file=Path(
            global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
        ),
    )
# ...
