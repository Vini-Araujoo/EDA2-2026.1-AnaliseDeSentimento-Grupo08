import spacy

# Load the small English model, disabling parser and NER for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
