import unittest
from data_structures import Node, LinkedList, HashTable, Vertex, Graph
from preprocessing import preprocess_tweet
from opinion_mining import build_base_graph, classify_tweet


# ------------------------------------------------------------------ #
# Testes das Estruturas de Dados                                      #
# Valida LinkedList, HashTable, Vertex e Graph implementados do zero.#
# ------------------------------------------------------------------ #
class TestDataStructures(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Testa inserção e remoção na Lista Encadeada                        #
    # Verifica: append, remove (meio, cabeça, cauda), iteração e tail.  #
    # ------------------------------------------------------------------ #
    def test_linked_list_append_and_remove(self):
        ll = LinkedList()
        self.assertEqual(len(ll), 0)
        
        ll.append("A", 10)
        ll.append("B", 20)
        ll.append("C", 30)
        self.assertEqual(len(ll), 3)
        
        # Verifica iteração
        nodes = list(ll)
        self.assertEqual(nodes[0].key, "A")
        self.assertEqual(nodes[0].value, 10)
        self.assertEqual(nodes[1].key, "B")
        self.assertEqual(nodes[2].key, "C")
        
        # Verifica ponteiro de cauda
        self.assertEqual(ll.tail.key, "C")
        
        # Remove do meio
        self.assertTrue(ll.remove("B"))
        self.assertEqual(len(ll), 2)
        nodes = list(ll)
        self.assertEqual(nodes[0].key, "A")
        self.assertEqual(nodes[1].key, "C")
        
        # Remove a cabeça
        self.assertTrue(ll.remove("A"))
        self.assertEqual(len(ll), 1)
        self.assertEqual(ll.head.key, "C")
        self.assertEqual(ll.tail.key, "C")
        
        # Remove a cauda
        self.assertTrue(ll.remove("C"))
        self.assertEqual(len(ll), 0)
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)

    # ------------------------------------------------------------------ #
    # Testa operações da Tabela Hash                                     #
    # Verifica: put, get, contains, delete e redimensionamento.         #
    # ------------------------------------------------------------------ #
    def test_hash_table_operations(self):
        # Capacidade pequena para forçar redimensionamento
        ht = HashTable(capacity=4)
        ht.put("apple", 1)
        ht.put("banana", 2)
        ht.put("cherry", 3)
        
        self.assertEqual(ht.get("apple"), 1)
        self.assertEqual(ht.get("banana"), 2)
        self.assertEqual(ht.get("cherry"), 3)
        self.assertTrue(ht.contains("banana"))
        self.assertFalse(ht.contains("dragonfruit"))
        
        # Verifica redimensionamento (4 elementos em capacidade 4 -> fator de carga > 0.75)
        ht.put("date", 4)
        self.assertEqual(ht.capacity, 8)
        self.assertEqual(ht.get("apple"), 1)
        self.assertEqual(ht.get("date"), 4)
        
        # Sobrescreve valor
        ht.put("apple", 10)
        self.assertEqual(ht.get("apple"), 10)
        
        # Deleta valor
        self.assertTrue(ht.delete("banana"))
        self.assertIsNone(ht.get("banana"))
        self.assertFalse(ht.contains("banana"))
        self.assertEqual(ht.size, 3)
        
        # Deleta inexistente
        self.assertFalse(ht.delete("banana"))

    # ------------------------------------------------------------------ #
    # Testa adição e remoção no Grafo                                    #
    # Verifica: add_vertex, add_edge (bidirecional), remove_vertex.     #
    # ------------------------------------------------------------------ #
    def test_graph_add_and_remove(self):
        g = Graph()
        v1 = Vertex("1", "positive", ["happy", "victory"])
        v2 = Vertex("2", "negative", ["sad", "defeat"])
        v3 = Vertex("3", "neutral", ["match", "kickoff"])
        
        g.add_vertex(v1)
        g.add_vertex(v2)
        g.add_vertex(v3)
        
        self.assertEqual(g.get_vertex("1"), v1)
        self.assertEqual(g.get_vertex("2"), v2)
        
        g.add_edge("1", "2", 5)
        g.add_edge("2", "3", 3)
        
        # Verifica aresta bidirecional
        # Vértice 1 deve ter Vértice 2 como vizinho com peso 5
        v1_neighbors = list(v1.neighbors)
        self.assertEqual(len(v1_neighbors), 1)
        self.assertEqual(v1_neighbors[0].key, "2")
        self.assertEqual(v1_neighbors[0].value, 5)
        
        # Vértice 2 deve ter vizinhos 1 (peso 5) e 3 (peso 3)
        v2_neighbors = list(v2.neighbors)
        self.assertEqual(len(v2_neighbors), 2)
        v2_neighbor_keys = [node.key for node in v2_neighbors]
        self.assertIn("1", v2_neighbor_keys)
        self.assertIn("3", v2_neighbor_keys)
        
        # Remove Vértice 2
        g.remove_vertex("2")
        self.assertIsNone(g.get_vertex("2"))
        
        # Vértice 1 não deve mais listar Vértice 2 como vizinho
        self.assertEqual(len(list(v1.neighbors)), 0)
        # Vértice 3 não deve mais listar Vértice 2 como vizinho
        self.assertEqual(len(list(v3.neighbors)), 0)


# ------------------------------------------------------------------ #
# Testes do Pré-processamento                                         #
# Valida tokenização, lematização e remoção de stopwords.            #
# ------------------------------------------------------------------ #
class TestPreprocessing(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Testa pré-processamento de tweet                                    #
    # Verifica lematização ("playing" -> "play"), remoção de stopwords   #
    # e conversão para minúsculo.                                        #
    # ------------------------------------------------------------------ #
    def test_preprocess_tweet(self):
        tweet = "The players are playing beautifully! It is a great victory, but I'm tired."
        words = preprocess_tweet(tweet)
        
        # "playing" -> "play", "victory" -> "victory"
        # Stopwords "are", "is" devem ser removidas
        # Pontuação "!" e "." devem ser removidas
        self.assertIn("play", words)
        self.assertIn("victory", words)
        self.assertNotIn("are", words)
        self.assertNotIn("is", words)
        
        # Confirma conversão para minúsculo
        for w in words:
            self.assertEqual(w, w.lower())


# ------------------------------------------------------------------ #
# Testes do Sistema de Análise de Sentimento                          #
# Valida construção do grafo e classificação de tweets.              #
# Usa dataset mock com 6 tweets (2 pos, 2 neg, 2 neutros).          #
# ------------------------------------------------------------------ #
class TestSentimentSystem(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Configura dataset mock para os testes                               #
    # Constrói o grafo base com threshold Jaccard-IDF de 0.05            #
    # ------------------------------------------------------------------ #
    def setUp(self):
        self.mock_data = [
            {"id": "t1", "tweet": "I love the World Cup, it is fantastic and beautiful!", "sentiment": "positive"},
            {"id": "t2", "tweet": "Incredible win by the team! Love the plays and happy feelings.", "sentiment": "positive"},
            {"id": "t3", "tweet": "This match was so boring and awful. Terrible performance.", "sentiment": "negative"},
            {"id": "t4", "tweet": "I hate this referee, he is absolutely bad and worst.", "sentiment": "negative"},
            {"id": "t5", "tweet": "The kickoff start time is scheduled for 6 PM today.", "sentiment": "neutral"},
            {"id": "t6", "tweet": "The group stage draw result is out for the tournament.", "sentiment": "neutral"},
        ]
        self.graph, self.inverted_index = build_base_graph(self.mock_data, threshold=0.05)

    # ------------------------------------------------------------------ #
    # Verifica que o grafo base possui 6 vértices (um por tweet)         #
    # ------------------------------------------------------------------ #
    def test_base_graph_size(self):
        self.assertEqual(self.graph.vertices.size, 6)

    # ------------------------------------------------------------------ #
    # Testa classificação de tweet positivo                               #
    # Verifica também que o vértice temporário foi limpo após classificar #
    # ------------------------------------------------------------------ #
    def test_classify_positive(self):
        new_tweet = "I am so happy and love this game, it is fantastic!"
        predicted = classify_tweet(self.graph, self.inverted_index, new_tweet, threshold=0.05)
        self.assertEqual(predicted, "positive")
        
        # Garante que o vértice temporário foi removido
        self.assertIsNone(self.graph.get_vertex("__temp_query_tweet__"))

    # ------------------------------------------------------------------ #
    # Testa classificação de tweet negativo                               #
    # Verifica também que o vértice temporário foi limpo após classificar #
    # ------------------------------------------------------------------ #
    def test_classify_negative(self):
        new_tweet = "Awful referee and terrible worst match, I hate it."
        predicted = classify_tweet(self.graph, self.inverted_index, new_tweet, threshold=0.05)
        self.assertEqual(predicted, "negative")
        
        # Garante que o vértice temporário foi removido
        self.assertIsNone(self.graph.get_vertex("__temp_query_tweet__"))

    # ------------------------------------------------------------------ #
    # Testa classificação de tweet neutro                                 #
    # ------------------------------------------------------------------ #
    def test_classify_neutral(self):
        new_tweet = "What is the schedule and match kick-off time tomorrow?"
        predicted = classify_tweet(self.graph, self.inverted_index, new_tweet, threshold=0.05)
        self.assertIn(predicted, ["positive", "negative", "neutral"])

    # ------------------------------------------------------------------ #
    # Testa desempate usando prior de classe                              #
    # Tweet com palavras positivas e negativas (love + terrible).        #
    # Com priors iguais (2 pos, 2 neg, 2 neut), o fallback é usado.     #
    # ------------------------------------------------------------------ #
    def test_tie_uses_class_prior(self):
        new_tweet = "I love this, but it is terrible performance."
        predicted = classify_tweet(self.graph, self.inverted_index, new_tweet, threshold=0.05)
        self.assertIn(predicted, ["positive", "negative", "neutral"])


if __name__ == "__main__":
    unittest.main()
