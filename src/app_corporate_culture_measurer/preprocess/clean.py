import datetime
import functools
import logging
import sys
from pathlib import Path

import pandas as pd

import global_options as global_options
import preprocess.parse as parse
from culture import file_util, preprocess

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def clean_file(in_file, out_file):
    """clean the entire corpus (output from CoreNLP)

    Arguments:
        in_file {str or Path} -- input corpus, each line is a sentence
        out_file {str or Path} -- output corpus
    """
    a_text_clearner = preprocess.text_cleaner()
    parse.process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=[
            str(i) for i in range(file_util.line_counter(in_file))
        ],  # fake IDs (do not need IDs for this function).
        output_index_file=None,
        function_name=functools.partial(a_text_clearner.clean),
        chunk_size=200000,
    )
