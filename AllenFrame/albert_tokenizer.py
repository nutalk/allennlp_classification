from typing import List, Optional
from overrides import overrides

import spacy

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from AllenFrame.albert.albert_total import get_albert_total


@Tokenizer.register("albert-basic")
class AlbertPreTokenizer(Tokenizer):
    """
    The ``BasicTokenizer`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """

    def __init__(self, config_path, vocab_path, model_path) -> None:
        config, tokenizer, model = get_albert_total(config_path, vocab_path, model_path)
        self.tokenizer = tokenizer

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        text = "[CLS] {} [SEP]".format(text)
        return [Token(text) for text in self.tokenizer.tokenize(text)]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]