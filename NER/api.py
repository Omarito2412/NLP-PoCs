import json
from dataclasses import dataclass
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import dacite

from ner import Extraction
from ner.transformer_ne_extractor import TransformerExtractor


@dataclass
class Configuration:
    model_name: str


class Input(BaseModel):
    text: str


class Output(BaseModel):
    named_entities: List[Extraction]


with open("config.json") as config_ptr:
    json_config = json.load(config_ptr)

config = dacite.from_dict(
    data_class=Configuration, data=json_config,
)

api = FastAPI()
named_entity_extractor = TransformerExtractor(config.model_name)


@api.post("/ner", response_model=Output)
def extractions(input_request: Input):
    ne_extractions = named_entity_extractor.extract(input_request.text)

    return {"named_entities": ne_extractions}
