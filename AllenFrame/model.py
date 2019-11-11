import logging

from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import modules

from .moduls import FocalLoss
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("text_classifier_fscore_focal_loss")
class BasicClassifierF1(Model):
    """
    文本分类模型。修改了fscore，focal loss等
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels: int = None,
            label_namespace: str = "labels",
            loss: str = None,  # focal_loss
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        if loss is None:
            self._loss = torch.nn.CrossEntropyLoss()
        elif loss == 'focal_loss':
            self._loss = FocalLoss(alpha=0.25, num_classes=self._num_labels)  # focal loss
        elif loss == 'cross_entropy_loss':
            self._loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('wrong loss type')
        self._f1_measure = FBetaMeasure()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1_measure(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_measure.get_metric(reset)
        output = {}
        output['accuracy'] = self._accuracy.get_metric(reset=reset)
        counter = 0
        for precision, recall, fscore in zip(f1_dict['precision'], f1_dict['recall'], f1_dict['fscore']):
            output[str(counter) + '_precision'] = precision
            output[str(counter) + '_recall'] = recall
            output[str(counter) + '_fscore'] = fscore
            counter += 1
        return output
