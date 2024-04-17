from stanfordnlp.server import CoreNLPClient
import global_options as global_options


class StandfordCoreNLPClient:
    client: CoreNLPClient = None

    def __init__(self) -> None:
        self.client = CoreNLPClient(
            properties={
                "ner.applyFineGrained": "false",
                "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
            },
            memory=global_options.RAM_CORENLP,
            threads=global_options.N_CORES,
            timeout=12000000,
            max_char_length=1000000,
        )
