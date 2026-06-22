import spacy

# ------------------------------------------------------------------ #
# Carrega o modelo pequeno de inglês do spaCy.                        #
# Desabilita parser e NER para ganho de velocidade.                   #
# ------------------------------------------------------------------ #
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ------------------------------------------------------------------ #
# Stopwords específicas do domínio (Copa do Mundo 2022).              #
# Palavras que não carregam sentimento e são muito comuns no dataset. #
# ------------------------------------------------------------------ #
DOMAIN_STOPWORDS = {
    'worldcup2022', 'qatar', 'world', 'cup', 'worldcup', 
    'fifaworldcup', 'qatar2022', 'football', 'fifa', 'match', 
    '2022', 'amp', 'qatarworldcup2022'
}


# ------------------------------------------------------------------ #
# Função preprocess_tweet                                              #
# Limpa o texto de um tweet usando spaCy.                             #
# Realiza tokenização, remoção de stopwords/pontuação e lematização.  #
# Também filtra stopwords específicas do domínio.                     #
# Retorna uma lista de lemas em minúsculo das palavras úteis.         #
# ------------------------------------------------------------------ #
def preprocess_tweet(text):
    if not isinstance(text, str):
        return []

    # Executa o pipeline do spaCy no texto
    doc = nlp(text)

    useful_words = []
    for token in doc:
        # Filtra stopwords, pontuação e tokens de espaço em branco
        if token.is_stop or token.is_punct or token.is_space:
            continue
        
        # Extrai o lema, converte para minúsculo e remove espaços
        lemma = token.lemma_.strip().lower()
        
        # Adiciona apenas lemas não-vazios que não são stopwords do domínio
        if lemma and lemma not in DOMAIN_STOPWORDS:
            useful_words.append(lemma)
            
    return useful_words
