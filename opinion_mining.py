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
