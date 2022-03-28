from functools import reduce
from typing import List, Tuple
import torch

from . import NeExtractor, Extraction, NeToken

from transformers import AutoTokenizer, AutoModelForTokenClassification


def groupby_bio(bio_predictions: List[NeToken]) -> List[Tuple[str, List[NeToken]]]:
    entities = []
    accumulator = []
    previous_tag = "O"
    for entity in bio_predictions:
        if entity.bio.replace("B-", "").replace("I-", "") == "O" and (
            previous_tag.startswith("I-") or previous_tag.startswith("B-")
        ):
            entities.append(
                (previous_tag.replace("B-", "").replace("I-", ""), accumulator)
            )
            accumulator = [entity]
            previous_tag = entity.bio
        if entity.bio.startswith("B-") and (
            previous_tag.startswith("I-") or previous_tag.startswith("B-")
        ):
            entities.append(
                (previous_tag.replace("B-", "").replace("I-", ""), accumulator)
            )
            accumulator = [entity]
            previous_tag = entity.bio
        if entity.bio.startswith("B-") and previous_tag.startswith("O"):
            accumulator = [entity]
            previous_tag = entity.bio
        if entity.bio.startswith("I-") and (
            previous_tag.startswith("B-") or previous_tag.startswith("I-")
        ):
            accumulator.append(entity)
            previous_tag = entity.bio

    if len(accumulator) > 0 and previous_tag.replace("B-", "").replace("I-", "") != "O":
        entities.append((previous_tag.replace("B-", "").replace("I-", ""), accumulator))
    return entities


class TransformerExtractor(NeExtractor):
    def __init__(self, model_name):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract(self, text: str) -> List[Extraction]:
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        tokens = inputs.tokens()
        offsets = inputs["offset_mapping"].squeeze().numpy()
        del inputs["offset_mapping"]
        outputs = torch.softmax(self.model(**inputs).logits, dim=2)

        bio_predictions = []

        for token, offset, output in zip(tokens, offsets, outputs[0]):
            if offset[0] == offset[1]:
                continue
            prediction = torch.argmax(output)
            bio_predictions.append(
                NeToken(
                    offset[0],
                    offset[1],
                    token,
                    self.model.config.id2label[prediction.numpy().item()],
                    output.max().item(),
                )
            )

        grouped_bio = groupby_bio(bio_predictions)

        extractions = []
        for named_entity in grouped_bio:
            start, end = named_entity[1][0].start, named_entity[1][-1].end
            if len(named_entity[1]) == 1:
                confidence = named_entity[1][0].confidence
            else:
                confidence = float(
                    reduce(lambda x, y: x + y, [n.confidence for n in named_entity[1]])
                ) / len(named_entity[1])
            extractions.append(
                Extraction(
                    start,
                    end,
                    named_entity[0],
                    text[start:end],
                    confidence,
                )
            )

        return extractions
