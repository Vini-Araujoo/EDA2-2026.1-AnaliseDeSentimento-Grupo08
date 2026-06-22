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
    for token in doc:
        # Filter out stopwords, punctuation, and whitespace tokens
        if token.is_stop or token.is_punct or token.is_space:
            continue
        
        # Extract the lemma, convert to lowercase, and strip whitespace
        lemma = token.lemma_.strip().lower()
        
        # Only add non-empty lemmas that are not domain-specific stopwords
        if lemma and lemma not in DOMAIN_STOPWORDS:
            useful_words.append(lemma)
            
    return useful_words
