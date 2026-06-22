import math
from data_structures import Vertex, Graph, HashTable, LinkedList
from preprocessing import preprocess_tweet


# ------------------------------------------------------------------ #
# Função build_base_graph                                             #
# Constrói o Grafo Base e o Índice Invertido a partir dos tweets.    #
#                                                                     #
# Melhorias em relação à versão original:                             #
# - Pesos das arestas usam Jaccard ponderado por IDF.                #
# - Tabela IDF calculada do corpus e armazenada no grafo.            #
# - Tabela class_prior para fallback/desempate na classificação.     #
#                                                                     #
# Parâmetros:                                                         #
# - tweets_data: iterável de dicts com 'id', 'tweet', 'sentiment'.  #
# - threshold: Jaccard-IDF mínimo para criar aresta (padrão 0.05).  #
#                                                                     #
# Retorna:                                                            #
# - graph: instância de Graph com tweets e arestas de similaridade.  #
# - inverted_index: HashTable mapeando palavras para IDs de tweets.  #
# ------------------------------------------------------------------ #
def build_base_graph(tweets_data, threshold=0.05):
    raw_list = list(tweets_data)
    N = len(raw_list)

    # ------------------------------------------------------------------ #
    # Passo 1: Calcular a frequência de documento (df) de cada palavra   #
    # df[w] = número de tweets que contêm a palavra w (cada tweet uma vez)#
    # ------------------------------------------------------------------ #
    word_df = {}
    preprocessed_tweets = []

    for item in raw_list:
        words = preprocess_tweet(item['tweet'])
        preprocessed_tweets.append(words)

        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                word_df[word] = word_df.get(word, 0) + 1

    # ------------------------------------------------------------------ #
    # Passo 2: Stopwords do corpus — palavras muito frequentes (> 1%)    #
    # que NÃO são palavras de sentimento conhecidas                       #
    # ------------------------------------------------------------------ #
    PROTECTED_WORDS = {
        'good', 'bad', 'great', 'terrible', 'awful', 'shitty', 'worst', 'best',
        'love', 'hate', 'happy', 'sad', 'win', 'lose', 'victory', 'defeat',
        'nice', 'perfect', 'amazing', 'fantastic', 'excellent', 'poor', 'well',
        'better', 'worse', 'boring', 'bored', 'fun', 'like', 'dislike', 'enjoy',
        'hope', 'proud', 'shame', 'disappoint', 'disappointed', 'disappointing',
        'mess', 'trash', 'garbage', 'cheat', 'referee', 'var', 'penalty', 'miss',
        'missed', 'lose', 'lost', 'won', 'winner', 'loser', 'beautiful'
    }
    cutoff = max(15, int(0.01 * N))
    corpus_stopwords = {
        w: True for w, df_w in word_df.items()
        if df_w > cutoff and w not in PROTECTED_WORDS
    }

    # ------------------------------------------------------------------ #
    # Passo 3: Construir tabela IDF                                       #
    # IDF(w) = log( (N + 1) / (df(w) + 1) ) + 1   (suavizado)          #
    # Palavras em muitos documentos recebem IDF baixo -> peso baixo.     #
    # Palavras em poucos documentos recebem IDF alto -> peso alto.       #
    # ------------------------------------------------------------------ #
    idf_table = {}
    for word, df_w in word_df.items():
        if word not in corpus_stopwords:
            idf_table[word] = math.log((N + 1) / (df_w + 1)) + 1.0

    # ------------------------------------------------------------------ #
    # Passo 4: Construir vértices do grafo, índice invertido e conjuntos #
    # de palavras por tweet                                               #
    # ------------------------------------------------------------------ #
    graph = Graph()
    graph.corpus_stopwords = corpus_stopwords
    graph.idf_table = idf_table
    graph.N_train = N

    inverted_index = HashTable(capacity=10000)

    # Mapeia tweet_id -> frozenset de palavras úteis para operações de conjunto
    word_sets = {}   # tweet_id (str) -> frozenset

    for i, item in enumerate(raw_list):
        tweet_id = item['id']
        sentiment = item['sentiment']

        useful_words = [w for w in preprocessed_tweets[i] if w not in corpus_stopwords]
        vertex = Vertex(tweet_id, sentiment, useful_words)
        graph.add_vertex(vertex)

        word_sets[tweet_id] = frozenset(useful_words)

        # Constrói índice invertido (palavras únicas por tweet)
        seen = set()
        for word in useful_words:
            if word not in seen:
                seen.add(word)
                postings = inverted_index.get(word)
                if postings is None:
                    postings = LinkedList()
                    inverted_index.put(word, postings)
                postings.append(tweet_id)

    # ------------------------------------------------------------------ #
    # Passo 5: Construir arestas usando Jaccard ponderado por IDF        #
    #                                                                     #
    # peso(A,B) = SOMA_IDF(A ∩ B)                                       #
    #             ────────────────                                        #
    #             SOMA_IDF(A ∪ B)                                        #
    #                                                                     #
    # Normaliza pelo tamanho do tweet E recompensa palavras raras.       #
    # ------------------------------------------------------------------ #
    all_ids = graph.vertices.keys()

    for tweet_id in all_ids:
        words_a = word_sets.get(tweet_id)
        if not words_a:
            continue

        # Usa índice invertido para encontrar candidatos eficientemente
        candidate_raw_counts = {}
        for word in words_a:
            postings = inverted_index.get(word)
            if postings:
                for node in postings:
                    cand_id = node.key
                    # Processa cada par uma vez: tweet_id < cand_id lexicograficamente
                    if cand_id != tweet_id and tweet_id < cand_id:
                        candidate_raw_counts[cand_id] = candidate_raw_counts.get(cand_id, 0) + 1

        for cand_id in candidate_raw_counts:
            words_b = word_sets.get(cand_id)
            if not words_b:
                continue

            intersection = words_a & words_b
            if not intersection:
                continue

            union = words_a | words_b
            idf_intersection = sum(idf_table.get(w, 1.0) for w in intersection)
            idf_union = sum(idf_table.get(w, 1.0) for w in union)

            jaccard_idf = idf_intersection / idf_union if idf_union > 0 else 0.0

            if jaccard_idf >= threshold:
                graph.add_edge(tweet_id, cand_id, jaccard_idf)

    # ------------------------------------------------------------------ #
    # Passo 6: Calcular e armazenar o prior de classe                    #
    # prior[sentimento] = contagem(sentimento) / N                       #
    # Usado como fallback quando não há vizinhos e para desempate.       #
    # ------------------------------------------------------------------ #
    class_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for item in raw_list:
        sent = item['sentiment'].lower().strip()
        if sent in class_counts:
            class_counts[sent] += 1

    graph.class_prior = {
        sent: count / N for sent, count in class_counts.items()
    }

    return graph, inverted_index

def classify_tweet(graph, inverted_index, text, threshold=1):
    """
    Classifies a new tweet into positive, negative, or neutral sentiment.

    Parameters:
    - graph: Base Graph containing the immutable dataset.
    - inverted_index: Inverted index mapping words to tweet IDs.
    - text: raw text of the new tweet.
    - threshold: minimum number of shared words to connect the new tweet to base tweets.

    Returns:
    - str: 'positive', 'negative', or 'neutral' (default on tie or no matches).
    """
    raw_words = preprocess_tweet(text)

    corpus_stopwords = getattr(graph, 'corpus_stopwords', {})
    useful_words = [w for w in raw_words if w not in corpus_stopwords]

    if not useful_words:
        return "neutral"

    new_tweet_id = "__temp_query_tweet__"
    candidate_counts = {}
    for word in useful_words:
        postings = inverted_index.get(word)
        if postings:
            for node in postings:
                cand_id = node.key
                candidate_counts[cand_id] = candidate_counts.get(cand_id, 0) + 1

    temp_vertex = Vertex(new_tweet_id, "unknown", useful_words)
    graph.add_vertex(temp_vertex)

    has_edges = False
    for cand_id, intersection in candidate_counts.items():
        if intersection >= threshold:
            graph.add_edge(new_tweet_id, cand_id, intersection)
            has_edges = True

    if not has_edges:
        graph.remove_vertex(new_tweet_id)
        return "neutral"

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    L1 = []
    L1_ids = {}
    for node in temp_vertex.neighbors:
        neighbor_id = node.key
        weight = node.value
        neighbor_vertex = graph.get_vertex(neighbor_id)
        if neighbor_vertex:
            L1.append((neighbor_vertex, weight))
            L1_ids[neighbor_id] = True

    for neighbor_vertex, weight in L1:
        sentiment = neighbor_vertex.sentiment.lower()
        if sentiment in scores:
            scores[sentiment] += weight

    for neighbor_vertex, w1 in L1:
        deg = len(neighbor_vertex.neighbors)
        if deg == 0:
            deg = 1
        for node in neighbor_vertex.neighbors:
            l2_id = node.key
            l2_weight = node.value

            if l2_id == new_tweet_id or l2_id in L1_ids:
                continue

            l2_vertex = graph.get_vertex(l2_id)
            if l2_vertex:
                sentiment = l2_vertex.sentiment.lower()
                if sentiment in scores:
                    scores[sentiment] += (0.5 * w1 * l2_weight) / deg

    winning_sentiment = "neutral"
    max_score = -1.0
    is_tie = False

    for sentiment, score in scores.items():
        if score > max_score:
            max_score = score
            winning_sentiment = sentiment
            is_tie = False
        elif score == max_score:
            is_tie = True

    if is_tie:
        final_sentiment = "neutral"
    else:
        final_sentiment = winning_sentiment

    graph.remove_vertex(new_tweet_id)

    return final_sentiment
