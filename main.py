import csv
import sys
import time
from opinion_mining import build_base_graph, classify_tweet

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