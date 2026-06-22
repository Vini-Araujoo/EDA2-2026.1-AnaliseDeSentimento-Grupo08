import spacy

# Load the small English model, disabling parser and NER for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Domain-specific stopwords for the 2022 World Cup dataset that do not carry sentiment
DOMAIN_STOPWORDS = {
    'worldcup2022', 'qatar', 'world', 'cup', 'worldcup', 
    'fifaworldcup', 'qatar2022', 'football', 'fifa', 'match', 
    '2022', 'amp', 'qatarworldcup2022'
}

def preprocess_tweet(text):
    """
    Cleans a tweet's text using spaCy.
    Performs tokenization, stopword and punctuation removal, and lemmatization.
    Also filters out domain-specific stopwords.
    Returns a list of lowercased lemmas of useful words.
    """
    if not isinstance(text, str):
        return []

    # Run the spaCy pipeline on the text
    doc = nlp(text)

    useful_words = []
