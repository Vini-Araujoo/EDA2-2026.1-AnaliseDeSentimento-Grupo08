from data_structures import Vertex, Graph, HashTable, LinkedList
from preprocessing import preprocess_tweet

def build_base_graph(tweets_data, threshold=1):
    """
    Builds the Base Graph and Inverted Index from the provided tweets data.
    
    Parameters:
    - tweets_data: iterable of dicts, each with keys 'id', 'tweet', and 'sentiment'.
    - threshold: minimum number of shared words to create an edge.
    
    Returns:
    - graph: Graph instance populated with tweets and similarity edges.
    - inverted_index: HashTable instance mapping words to LinkedList of tweet IDs.
    """
    raw_list = list(tweets_data)
    
    word_counts = {}
    preprocessed_tweets = []
    for item in raw_list:
        words = preprocess_tweet(item['tweet'])
        preprocessed_tweets.append(words)
        
        seen_words = {}
        for word in words:
            if word not in seen_words:
                seen_words[word] = True
                word_counts[word] = word_counts.get(word, 0) + 1

    PROTECTED_WORDS = {
        'good', 'bad', 'great', 'terrible', 'awful', 'shitty', 'worst', 'best',
        'love', 'hate', 'happy', 'sad', 'win', 'lose', 'victory', 'defeat',
        'nice', 'perfect', 'amazing', 'fantastic', 'excellent', 'poor', 'well',
        'better', 'worse', 'boring', 'bored', 'fun', 'like', 'dislike', 'enjoy',
        'hope', 'proud', 'shame', 'disappoint', 'disappointed', 'disappointing',
        'mess', 'trash', 'garbage', 'cheat', 'referee', 'var', 'penalty', 'miss',
        'missed', 'lose', 'lost', 'won', 'winner', 'loser', 'beautiful'
    }
    N = len(raw_list)
    cutoff = max(15, int(0.01 * N))
    corpus_stopwords = {w: True for w, c in word_counts.items() if c > cutoff and w not in PROTECTED_WORDS}

    graph = Graph()
    graph.corpus_stopwords = corpus_stopwords
    inverted_index = HashTable(capacity=5000)

    for i, item in enumerate(raw_list):
        tweet_id = item['id']
        sentiment = item['sentiment']
        
        useful_words = [w for w in preprocessed_tweets[i] if w not in corpus_stopwords]
        
        vertex = Vertex(tweet_id, sentiment, useful_words)
        graph.add_vertex(vertex)
        
        seen_words = {}
        for word in useful_words:
            if word not in seen_words:
                seen_words[word] = True
                postings = inverted_index.get(word)
                if postings is None:
                    postings = LinkedList()
                    inverted_index.put(word, postings)
                postings.append(tweet_id)

    all_ids = graph.vertices.keys()
    for tweet_id in all_ids:
        vertex = graph.get_vertex(tweet_id)
        if not vertex:
            continue
            
        candidate_counts = {}
        for word in vertex.useful_words:
            postings = inverted_index.get(word)
            if postings:
                for node in postings:
                    cand_id = node.key
                    if cand_id != tweet_id and tweet_id < cand_id:
                        candidate_counts[cand_id] = candidate_counts.get(cand_id, 0) + 1
                        
        for cand_id, count in candidate_counts.items():
            if count >= threshold:
                graph.add_edge(tweet_id, cand_id, count)
                
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

