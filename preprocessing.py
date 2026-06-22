import spacy

# Load the small English model, disabling parser and NER for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Domain-specific stopwords for the 2022 World Cup dataset that do not carry sentiment
DOMAIN_STOPWORDS = {
    'worldcup2022', 'qatar', 'world', 'cup', 'worldcup', 
    'fifaworldcup', 'qatar2022', 'football', 'fifa', 'match', 
    '2022', 'amp', 'qatarworldcup2022'
}
