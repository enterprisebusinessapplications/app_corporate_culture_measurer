import logging
import sys
from logging import log
from pathlib import Path

from preprocess.parse import sequential_parse
from preprocess.clean import clean_file
from train.train import train
import global_options

# from train.train import sequential_parse
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

if __name__ == "__main__":
    preprocess = False
    if(preprocess):
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
    train()
# ...
