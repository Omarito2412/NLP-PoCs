from dataclasses import dataclass

import streamlit as st
import pandas as pd
from ner import Extraction
from ner.transformer_ne_extractor import TransformerExtractor
import dacite
import json

st.title("NER Demo")
st.header("Extract Named Entities from text")

with open("config.json") as config_ptr:
    json_config = json.load(config_ptr)


@dataclass
class Configuration:
    model_name: str


config = dacite.from_dict(
    data_class=Configuration,
    data=json_config,
)


@st.cache(allow_output_mutation=True)
def load_ner_model(model_name: str):
    named_entity_extractor = TransformerExtractor(model_name)
    return named_entity_extractor


model = load_ner_model(config.model_name)
text = st.text_area(
    "Text to extract from",
    "Yann Lecun is a very famous scientist who works at Meta AI.",
    max_chars=500,
)

ne_extractions = model.extract(text)
predictions = pd.DataFrame(
    data={
        "text": [e.text for e in ne_extractions],
        "type": [e.type for e in ne_extractions],
    }
)
st.write(predictions)
