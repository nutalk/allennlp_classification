from typing import List

from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer
from overrides import overrides
from AllenFrame.albert.tokenization_bert import BertTokenizer as AlbertTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
import logging
logger = logging.getLogger(__name__)

@TokenIndexer.register("albert-pretrained")
class PretrainedAlbertIndexer(WordpieceIndexer):
    """

    """
    def __init__(
        self,
        pretrained_model: str,
        use_starting_offsets: bool = False,
        do_lowercase: bool = True,
        never_lowercase: List[str] = None,
        max_pieces: int = 512,
        truncate_long_sequences: bool = True,
    ) -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning(
                "Your BERT model appears to be cased, " "but your indexer is lowercasing tokens."
            )
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning(
                "Your BERT model appears to be uncased, "
                "but your indexer is not lowercasing tokens."
            )

        bert_tokenizer = AlbertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
        super().__init__(
            vocab=bert_tokenizer.vocab,
            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            namespace="albert",
            use_starting_offsets=use_starting_offsets,
            max_pieces=max_pieces,
            do_lowercase=do_lowercase,
            never_lowercase=never_lowercase,
            start_tokens=["[CLS]"],
            end_tokens=["[SEP]"],
            separator_token="[SEP]",
            truncate_long_sequences=truncate_long_sequences,
        )

    def __eq__(self, other):
        if isinstance(other, PretrainedAlbertIndexer):
            for key in self.__dict__:
                if key == "wordpiece_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented