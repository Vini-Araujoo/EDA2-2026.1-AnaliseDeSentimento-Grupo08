import csv
import sys
 
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
