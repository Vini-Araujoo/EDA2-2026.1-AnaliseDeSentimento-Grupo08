import csv
import time
import sys
from opinion_mining import build_base_graph, classify_tweet, preprocess_tweet
from data_structures import Vertex

def load_dataset(file_path, limit=6000):
    """
    Loads tweets from the CSV file.
    
    Returns:
    - list of dicts, each with keys 'id', 'tweet', 'sentiment'
    """
    tweets = []
    print(f"[*] Loading raw dataset from {file_path}...")
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                tweets.append({
                    'id': str(i),
                    'tweet': row['Tweet'],
                    'sentiment': row['Sentiment'].lower().strip()
                })
    except FileNotFoundError:
        print(f"[!] Error: {file_path} not found. Please place it in the same directory.")
        sys.exit(1)
    print(f"[+] Loaded {len(tweets)} tweets.")
    return tweets


def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics (Accuracy, Precision, Recall, F1-Score) from scratch.
    """
    classes = ['positive', 'negative', 'neutral']
    metrics = {}
    
    # Calculate overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    
    for cls in classes:
        tp = 0
        fp = 0
        fn = 0
        
        for t, p in zip(y_true, y_pred):
            if t == cls and p == cls:
                tp += 1
            elif t != cls and p == cls:
                fp += 1
            elif t == cls and p != cls:
                fn += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': tp + fn
        }
        
    return accuracy, metrics


def correct_count(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p)


def run_evaluation(tweets, train_size=4000, test_size=1000, threshold=1):
    """
    Splits the dataset, builds the Base Graph, and evaluates performance on a test set.
    """
    if len(tweets) < (train_size + test_size):
        print(f"[!] Warning: Dataset size ({len(tweets)}) is less than requested split size ({train_size + test_size}). Adjusting sizes.")
        train_size = int(len(tweets) * 0.8)
        test_size = len(tweets) - train_size

    train_data = tweets[:train_size]
    test_data = tweets[train_size:train_size + test_size]

    print(f"\n[*] Building Base Graph with {train_size} tweets (threshold={threshold})...")
    start_time = time.time()
    graph, inverted_index = build_base_graph(train_data, threshold=threshold)
    build_time = time.time() - start_time
    print(f"[+] Graph built in {build_time:.2f} seconds.")
    
    # Count total edges in base graph
    total_edges = 0
    all_ids = graph.vertices.keys()
    for tweet_id in all_ids:
        v = graph.get_vertex(tweet_id)
        if v:
            total_edges += len(v.neighbors)
    # Since undirected, each edge is counted twice
    total_edges //= 2
    print(f"[+] Total vertices: {graph.vertices.size}, Total edges: {total_edges}")

    print(f"\n[*] Evaluating on {test_size} test tweets...")
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for idx, item in enumerate(test_data):
        y_true.append(item['sentiment'])
        pred = classify_tweet(graph, inverted_index, item['tweet'], threshold=threshold)
        y_pred.append(pred)
        if (idx + 1) % 200 == 0:
            print(f"    Processed {idx + 1}/{test_size}...")
            
    eval_time = time.time() - start_time
    avg_speed = test_size / eval_time if eval_time > 0 else 0
    print(f"[+] Evaluation finished in {eval_time:.2f} seconds ({avg_speed:.1f} tweets/sec).")

    accuracy, metrics = calculate_metrics(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f" EVALUATION REPORT (threshold={threshold})")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_count(y_true, y_pred)}/{len(y_true)})")
    print("-"*50)
    print(f"{'Sentiment':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-"*50)
    for cls, scores in metrics.items():
        print(f"{cls:<12} | {scores['precision']:<10.4f} | {scores['recall']:<10.4f} | {scores['f1-score']:<10.4f} | {scores['support']:<8}")
    print("="*50 + "\n")


def interactive_trace(graph, inverted_index, text, threshold=1):
    """
    Runs classification on a single input text and displays the trace details.
    """
    print("\n" + "-"*60)
    print(" INTERACTIVE CLASSIFICATION TRACE")
    print("-"*60)
    
    print(f"Input text: \"{text}\"")
    
    # Preprocessing
    raw_words = preprocess_tweet(text)
    corpus_stopwords = getattr(graph, 'corpus_stopwords', {})
    words = [w for w in raw_words if w not in corpus_stopwords]
    print(f"Extracted Useful Words: {words}")
    
    if not words:
        print("[!] No useful words found (filtered as stopwords/punctuation).")
        print("Final Classification: NEUTRAL (default)")
        print("-"*60 + "\n")
        return

    # Candidate Search
    print("\n[Step 1] Querying Inverted Index...")
    candidate_counts = {}
    for word in words:
        postings = inverted_index.get(word)
        if postings:
            posting_ids = [node.key for node in postings]
            print(f"  Word '{word}' appears in: {posting_ids[:10]}... (Total: {len(posting_ids)})")
            for node in postings:
                cand_id = node.key
                candidate_counts[cand_id] = candidate_counts.get(cand_id, 0) + 1
        else:
            print(f"  Word '{word}' not found in index.")

    # Edge creation
    print("\n[Step 2] Calculating Intersection Weights (Threshold >= {})...".format(threshold))
    temp_id = "__temp_query_tweet__"
    temp_vertex = Vertex(temp_id, "unknown", words)
    graph.add_vertex(temp_vertex)
    
    edges_added = []
    for cand_id, intersection in candidate_counts.items():
        if intersection >= threshold:
            graph.add_edge(temp_id, cand_id, intersection)
            cand_vertex = graph.get_vertex(cand_id)
            edges_added.append((cand_id, cand_vertex.sentiment, cand_vertex.useful_words, intersection))
            
    if not edges_added:
        print("  [!] No candidates met the similarity threshold.")
        print("Final Classification: NEUTRAL (default)")
        graph.remove_vertex(temp_id)
        print("-"*60 + "\n")
        return
        
    print(f"  Connected to {len(edges_added)} direct neighbors in the Base Graph:")
    for cid, sent, useful, wt in edges_added[:10]:
        print(f"    - Base Tweet ID {cid} ({sent}, words: {useful}) -> Weight: {wt}")
    if len(edges_added) > 10:
        print(f"    - ... and {len(edges_added) - 10} more.")

    # Level-2 BFS scores calculation
    print("\n[Step 3] Propagating Scores (Level-2 BFS)...")
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    
    # Level 1 neighbors
    L1 = []
    L1_ids = {}
    print("  Level 1 Contributions (Direct Neighbors):")
    for node in temp_vertex.neighbors:
        neighbor_id = node.key
        weight = node.value
        neighbor_vertex = graph.get_vertex(neighbor_id)
        if neighbor_vertex:
            L1.append((neighbor_vertex, weight))
            L1_ids[neighbor_id] = True
            scores[neighbor_vertex.sentiment.lower()] += weight
            print(f"    Neighbor ID {neighbor_id} ({neighbor_vertex.sentiment}) -> Added weight: {weight}")

    # Level 2 neighbors
    print("\n  Level 2 Contributions (Neighbors of Neighbors, 0.5x penalty):")
    for neighbor_vertex, w1 in L1:
        print(f"    Exploring neighbors of Level 1 node {neighbor_vertex.id} ({neighbor_vertex.sentiment}):")
        deg = len(neighbor_vertex.neighbors)
        if deg == 0:
            deg = 1
        has_l2 = False
        for node in neighbor_vertex.neighbors:
            l2_id = node.key
            l2_weight = node.value
            
            if l2_id == temp_id or l2_id in L1_ids:
                continue
                
            l2_vertex = graph.get_vertex(l2_id)
            if l2_vertex:
                contrib = (0.5 * w1 * l2_weight) / deg
                scores[l2_vertex.sentiment.lower()] += contrib
                has_l2 = True
                print(f"      -> Neighbor ID {l2_id} ({l2_vertex.sentiment}) with edge weight {l2_weight} -> Added weight: {contrib:.4f}")
        if not has_l2:
            print("      -> No eligible Level 2 neighbors.")

    print("\n[Step 4] Final Scores Summary:")
    for sent, val in scores.items():
        print(f"  - {sent.capitalize()}: {val:.2f}")

    # Final Decision
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
        print("  Result: TIE detected! Defaulting to Neutral.")
        final_sentiment = "neutral"
    else:
        final_sentiment = winning_sentiment
        
    print(f"Final Classification Decision: {final_sentiment.upper()}")
    
    # Cleanup
    graph.remove_vertex(temp_id)
    print("-"*60 + "\n")


def main():
    csv_file = 'fifa_world_cup_2022_tweets.csv'
    
    # Check if a sentence was passed via command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h', '--help'):
            print("Usage:")
            print("  python main.py                    # Runs the interactive menu")
            print("  python main.py \"your tweet text\" # Directly runs a detailed classification trace")
            return
            
        tweet_text = " ".join(sys.argv[1:])
        tweets = load_dataset(csv_file, limit=5000)
        print("\n[*] Building base graph of 5000 tweets for query classification...")
        start_time = time.time()
        graph, inverted_index = build_base_graph(tweets, threshold=1)
        print(f"[+] Graph built in {time.time() - start_time:.2f} seconds.")
        
        interactive_trace(graph, inverted_index, tweet_text, threshold=1)
        return

    # Build a small initial graph for interactive use
    tweets = load_dataset(csv_file, limit=10000)
    
    print("\nOptions:")
    print("1. Run Full System Evaluation (Train size: 4000, Test size: 1000)")
    print("2. Run Interactive Mode (Input your own tweets)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice == '1':
        try:
            thresh_str = input("Enter similarity threshold (default: 1): ").strip()
            threshold = int(thresh_str) if thresh_str else 1
        except ValueError:
            print("[!] Invalid input. Using default threshold 1.")
            threshold = 1
        
        run_evaluation(tweets, train_size=4000, test_size=1000, threshold=threshold)
        
    elif choice == '2':
        # For interactive mode, we will build a base graph of 5000 tweets to query against
        print("\n[*] Building base graph of 5000 tweets for interactive queries...")
        start_time = time.time()
        graph, inverted_index = build_base_graph(tweets[:5000], threshold=1)
        print(f"[+] Graph built in {time.time() - start_time:.2f} seconds.")
        
        try:
            thresh_str = input("Enter query similarity threshold (default: 1): ").strip()
            threshold = int(thresh_str) if thresh_str else 1
        except ValueError:
            print("[!] Invalid input. Using default threshold 1.")
            threshold = 1
            
        print("\nEntering interactive mode. Type 'exit' or press Ctrl+C to stop.")
        while True:
            try:
                tweet_text = input("\nEnter tweet text to classify: ").strip()
                if not tweet_text:
                    continue
                if tweet_text.lower() == 'exit':
                    break
                interactive_trace(graph, inverted_index, tweet_text, threshold=threshold)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
    else:
        print("Exiting.")

if __name__ == '__main__':
    main()