import csv
import time
import sys
from opinion_mining import build_base_graph, classify_tweet, preprocess_tweet
from data_structures import Vertex

# ------------------------------------------------------------------ #
# Função load_dataset                                                 #
# Carrega tweets do arquivo CSV.                                      #
# Retorna lista de dicts com chaves 'id', 'tweet', 'sentiment'.      #
# ------------------------------------------------------------------ #
def load_dataset(file_path, limit=10000):
    tweets = []
    print(f"[*] Carregando dataset de {file_path}...")
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
        print(f"[!] Erro: {file_path} não encontrado. Coloque-o no mesmo diretório.")
        sys.exit(1)
    print(f"[+] {len(tweets)} tweets carregados.")
    return tweets


# ------------------------------------------------------------------ #
# Função calculate_metrics                                            #
# Calcula métricas de classificação do zero:                         #
# Acurácia, Precisão, Recall e F1-Score por classe.                  #
# ------------------------------------------------------------------ #
def calculate_metrics(y_true, y_pred):
    classes = ['positive', 'negative', 'neutral']
    metrics = {}
    
    # ------------------------------------------------------------------ #
    # Calcula a acurácia geral                                            #
    # acurácia = acertos / total                                          #
    # ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
# Função correct_count                                                #
# Conta o número de predições corretas.                              #
# ------------------------------------------------------------------ #
def correct_count(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p)


# ------------------------------------------------------------------ #
# Função run_evaluation                                               #
# Divide o dataset, constrói o Grafo Base e avalia a performance     #
# no conjunto de teste.                                               #
# ------------------------------------------------------------------ #
def run_evaluation(tweets, train_size=8000, test_size=2000, threshold=0.05):
    if len(tweets) < (train_size + test_size):
        print(f"[!] Aviso: Tamanho do dataset ({len(tweets)}) menor que o split solicitado ({train_size + test_size}). Ajustando tamanhos.")
        train_size = int(len(tweets) * 0.8)
        test_size = len(tweets) - train_size

    train_data = tweets[:train_size]
    test_data = tweets[train_size:train_size + test_size]

    print(f"\n[*] Construindo Grafo Base com {train_size} tweets (threshold={threshold})...")
    start_time = time.time()
    graph, inverted_index = build_base_graph(train_data, threshold=threshold)
    build_time = time.time() - start_time
    print(f"[+] Grafo construído em {build_time:.2f} segundos.")
    
    # ------------------------------------------------------------------ #
    # Conta o total de arestas no grafo base                              #
    # Como o grafo é não-direcionado, cada aresta é contada duas vezes   #
    # ------------------------------------------------------------------ #
    total_edges = 0
    all_ids = graph.vertices.keys()
    for tweet_id in all_ids:
        v = graph.get_vertex(tweet_id)
        if v:
            total_edges += len(v.neighbors)
    total_edges //= 2
    print(f"[+] Total de vértices: {graph.vertices.size}, Total de arestas: {total_edges}")

    print(f"\n[*] Avaliando em {test_size} tweets de teste...")
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for idx, item in enumerate(test_data):
        y_true.append(item['sentiment'])
        pred = classify_tweet(graph, inverted_index, item['tweet'], threshold=threshold)
        y_pred.append(pred)
        if (idx + 1) % 200 == 0:
            print(f"    Processados {idx + 1}/{test_size}...")
            
    eval_time = time.time() - start_time
    avg_speed = test_size / eval_time if eval_time > 0 else 0
    print(f"[+] Avaliação finalizada em {eval_time:.2f} segundos ({avg_speed:.1f} tweets/seg).")

    accuracy, metrics = calculate_metrics(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f" RELATÓRIO DE AVALIAÇÃO (threshold={threshold})")
    print("="*50)
    print(f"Acurácia Geral: {accuracy:.4f} ({correct_count(y_true, y_pred)}/{len(y_true)})")
    print("-"*50)
    print(f"{'Sentimento':<12} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-"*50)
    for cls, scores in metrics.items():
        print(f"{cls:<12} | {scores['precision']:<10.4f} | {scores['recall']:<10.4f} | {scores['f1-score']:<10.4f} | {scores['support']:<8}")
    print("="*50 + "\n")


# ------------------------------------------------------------------ #
# Função interactive_trace                                            #
# Executa a classificação em um texto e exibe os detalhes do trace.  #
# Mostra scores Jaccard-IDF e uso do prior de classe.                #
# ------------------------------------------------------------------ #
def interactive_trace(graph, inverted_index, text, threshold=0.05):
    import math
    print("\n" + "-"*60)
    print(" TRACE DE CLASSIFICAÇÃO INTERATIVA")
    print("-"*60)

    print(f"Texto de entrada: \"{text}\"")

    # ------------------------------------------------------------------ #
    # Pré-processamento do texto de entrada                               #
    # ------------------------------------------------------------------ #
    raw_words = preprocess_tweet(text)
    corpus_stopwords = getattr(graph, 'corpus_stopwords', {})
    idf_table = getattr(graph, 'idf_table', {})
    class_prior = getattr(graph, 'class_prior',
                          {'positive': 0.333, 'negative': 0.333, 'neutral': 0.334})

    words = [w for w in raw_words if w not in corpus_stopwords]
    print(f"Palavras Úteis Extraídas: {words}")

    if not words:
        fallback = max(class_prior, key=class_prior.get)
        print("[!] Nenhuma palavra útil encontrada (filtradas como stopwords/pontuação).")
        print(f"[Prior] Distribuição de classes: { {s: f'{p:.3f}' for s, p in class_prior.items()} }")
        print(f"Classificação Final: {fallback.upper()} (fallback por prior de classe)")
        print("-"*60 + "\n")
        return

    query_set = frozenset(words)

    # ------------------------------------------------------------------ #
    # Passo 1: Consultar o Índice Invertido                               #
    # Busca quais tweets do grafo base contêm as palavras da query       #
    # ------------------------------------------------------------------ #
    print("\n[Passo 1] Consultando Índice Invertido...")
    candidate_set = set()
    for word in query_set:
        postings = inverted_index.get(word)
        if postings:
            posting_ids = [node.key for node in postings]
            print(f"  Palavra '{word}' (IDF={idf_table.get(word, 1.0):.3f}) encontrada em {len(posting_ids)} tweets")
            for node in postings:
                candidate_set.add(node.key)
        else:
            print(f"  Palavra '{word}' não encontrada no índice.")

    # ------------------------------------------------------------------ #
    # Passo 2: Calcular similaridade Jaccard ponderada por IDF           #
    # Cria arestas temporárias entre a query e candidatos acima do limiar #
    # ------------------------------------------------------------------ #
    print(f"\n[Passo 2] Calculando Jaccard-IDF (threshold >= {threshold})...")
    temp_id = "__temp_query_tweet__"
    temp_vertex = Vertex(temp_id, "unknown", list(words))
    graph.add_vertex(temp_vertex)

    edges_added = []
    for cand_id in candidate_set:
        cand_vertex = graph.get_vertex(cand_id)
        if not cand_vertex:
            continue
        words_b = frozenset(cand_vertex.useful_words)
        intersection = query_set & words_b
        if not intersection:
            continue
        union = query_set | words_b
        idf_num = sum(idf_table.get(w, 1.0) for w in intersection)
        idf_den = sum(idf_table.get(w, 1.0) for w in union)
        jaccard_idf = idf_num / idf_den if idf_den > 0 else 0.0
        if jaccard_idf >= threshold:
            graph.add_edge(temp_id, cand_id, jaccard_idf)
            edges_added.append((cand_id, cand_vertex.sentiment, list(intersection), jaccard_idf))

    if not edges_added:
        fallback = max(class_prior, key=class_prior.get)
        print("  [!] Nenhum candidato atingiu o limiar de similaridade.")
        print(f"[Prior] Distribuição de classes: { {s: f'{p:.3f}' for s, p in class_prior.items()} }")
        print(f"Classificação Final: {fallback.upper()} (fallback por prior de classe)")
        graph.remove_vertex(temp_id)
        print("-"*60 + "\n")
        return

    print(f"  Conectado a {len(edges_added)} vizinhos:")
    for cid, sent, common, jac in sorted(edges_added, key=lambda x: -x[3])[:10]:
        print(f"    - ID {cid} ({sent}) | compartilhadas={common} | Jaccard-IDF={jac:.4f}")
    if len(edges_added) > 10:
        print(f"    - ... e mais {len(edges_added) - 10}.")

    # ------------------------------------------------------------------ #
    # Passo 3: Propagação de scores (BFS Nível 2 com pesos normalizados) #
    # L1: vizinhos diretos contribuem com peso Jaccard-IDF               #
    # L2: vizinhos dos vizinhos com decaimento de 0.5                    #
    # ------------------------------------------------------------------ #
    print("\n[Passo 3] Propagando Scores (BFS Nível 2 com pesos normalizados)...")
    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    L1 = []
    L1_ids = set()
    print("  Contribuições Nível 1 (Vizinhos Diretos):")
    for node in temp_vertex.neighbors:
        neighbor_id = node.key
        weight = node.value
        neighbor_vertex = graph.get_vertex(neighbor_id)
        if neighbor_vertex:
            L1.append((neighbor_vertex, weight))
            L1_ids.add(neighbor_id)
            scores[neighbor_vertex.sentiment.lower()] += weight
            print(f"    Vizinho ID {neighbor_id} ({neighbor_vertex.sentiment}) -> Jaccard-IDF: {weight:.4f}")

    max_l1_weight = max((w for _, w in L1), default=1.0) or 1.0

    print("\n  Contribuições Nível 2 (Vizinhos dos Vizinhos, decaimento=0.5):")
    L2_DECAY = 0.5
    for neighbor_vertex, w1 in L1:
        deg = len(neighbor_vertex.neighbors) or 1
        norm_w1 = w1 / max_l1_weight
        has_l2 = False
        for node in neighbor_vertex.neighbors:
            l2_id = node.key
            l2_weight = node.value
            if l2_id == temp_id or l2_id in L1_ids:
                continue
            l2_vertex = graph.get_vertex(l2_id)
            if l2_vertex:
                contrib = L2_DECAY * norm_w1 * (l2_weight / deg)
                scores[l2_vertex.sentiment.lower()] += contrib
                has_l2 = True
                print(f"      -> ID {l2_id} ({l2_vertex.sentiment}) | w={l2_weight:.4f} | contrib={contrib:.5f}")
        if not has_l2:
            print(f"      -> Nó L1 {neighbor_vertex.id}: nenhum vizinho Nível 2 elegível.")

    # ------------------------------------------------------------------ #
    # Passo 4: Scores finais e decisão                                    #
    # Em caso de empate ou score zero, usa o prior de classe como fallback#
    # ------------------------------------------------------------------ #
    print("\n[Passo 4] Scores Finais:")
    for sent, val in scores.items():
        print(f"  - {sent.capitalize()}: {val:.4f}")

    winning_sentiment = None
    max_score = -1.0
    is_tie = False

    for sentiment, score in scores.items():
        if score > max_score:
            max_score = score
            winning_sentiment = sentiment
            is_tie = False
        elif score == max_score and score > 0.0:
            is_tie = True

    if is_tie or max_score <= 0.0:
        final_sentiment = max(class_prior, key=class_prior.get)
        print(f"  [Empate/Zero] Usando prior de classe: { {s: f'{p:.3f}' for s, p in class_prior.items()} }")
    else:
        final_sentiment = winning_sentiment

    print(f"Decisão Final de Classificação: {final_sentiment.upper()}")

    # Limpeza do vértice temporário
    graph.remove_vertex(temp_id)
    print("-"*60 + "\n")


def main():
    csv_file = 'fifa_world_cup_2022_tweets.csv'
    
    # Verifica se uma frase foi passada via argumentos de linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h', '--help'):
            print("Uso:")
            print("  python main.py                   # Executa o menu interativo")
            print("  python main.py \"texto do tweet\" # Executa trace detalhado de classificação")
            return
            
        tweet_text = " ".join(sys.argv[1:])
        tweets = load_dataset(csv_file, limit=10000)
        print("\n[*] Construindo grafo base de 10000 tweets para classificação...")
        start_time = time.time()
        graph, inverted_index = build_base_graph(tweets, threshold=0.05)
        print(f"[+] Grafo construído em {time.time() - start_time:.2f} segundos.")
        
        interactive_trace(graph, inverted_index, tweet_text, threshold=0.05)
        return

    # Carrega dataset para uso interativo
    tweets = load_dataset(csv_file, limit=10000)
    
    print("\nOpções:")
    print("1. Executar Avaliação Completa (Treino: 8000, Teste: 2000)")
    print("2. Executar Modo Interativo (Digite seus próprios tweets)")
    print("3. Sair")
    
    try:
        choice = input("\nEscolha (1-3): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSaindo.")
        return

    if choice == '1':
        try:
            thresh_str = input("Digite o threshold de similaridade Jaccard-IDF (padrão: 0.05): ").strip()
            threshold = float(thresh_str) if thresh_str else 0.05
        except ValueError:
            print("[!] Entrada inválida. Usando threshold padrão 0.05.")
            threshold = 0.05
        
        run_evaluation(tweets, train_size=8000, test_size=2000, threshold=threshold)
        
    elif choice == '2':
        # Para modo interativo, constrói grafo base de 10000 tweets
        print("\n[*] Construindo grafo base de 10000 tweets para consultas interativas...")
        start_time = time.time()
        graph, inverted_index = build_base_graph(tweets[:10000], threshold=0.05)
        print(f"[+] Grafo construído em {time.time() - start_time:.2f} segundos.")

        try:
            thresh_str = input("Digite o threshold de similaridade Jaccard-IDF (padrão: 0.05): ").strip()
            threshold = float(thresh_str) if thresh_str else 0.05
        except ValueError:
            print("[!] Entrada inválida. Usando threshold padrão 0.05.")
            threshold = 0.05
            
        print("\nEntrando no modo interativo. Digite 'exit' ou Ctrl+C para sair.")
        while True:
            try:
                tweet_text = input("\nDigite o texto do tweet para classificar: ").strip()
                if not tweet_text:
                    continue
                if tweet_text.lower() == 'exit':
                    break
                interactive_trace(graph, inverted_index, tweet_text, threshold=threshold)
            except (KeyboardInterrupt, EOFError):
                print("\nSaindo do modo interativo.")
                break
    else:
        print("Saindo.")

if __name__ == '__main__':
    main()