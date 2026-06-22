import unittest
from hypothesis import given, strategies as st, settings
from data_structures import Node, LinkedList, HashTable, Vertex, Graph
from preprocessing import preprocess_tweet, DOMAIN_STOPWORDS
from opinion_mining import build_base_graph, classify_tweet


# ------------------------------------------------------------------ #
# Testes do LinkedList com Hypothesis                                 #
# ------------------------------------------------------------------ #
class TestHypothesisLinkedList(unittest.TestCase):

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers())))
    def test_append_and_iteration(self, items):
        ll = LinkedList()
        self.assertEqual(len(ll), 0)
        for k, v in items:
            ll.append(k, v)
        self.assertEqual(len(ll), len(items))
        
        # ------------------------------------------------------------------ #
        # Verifica se a iteração retorna os mesmos elementos e ordem         #
        # ------------------------------------------------------------------ #
        ll_nodes = list(ll)
        self.assertEqual(len(ll_nodes), len(items))
        for node, (k, v) in zip(ll_nodes, items):
            self.assertEqual(node.key, k)
            self.assertEqual(node.value, v)
            
        if items:
            self.assertEqual(ll.tail.key, items[-1][0])
            self.assertEqual(ll.tail.value, items[-1][1])
            self.assertEqual(ll.head.key, items[0][0])
            self.assertEqual(ll.head.value, items[0][1])
        else:
            self.assertIsNone(ll.head)
            self.assertIsNone(ll.tail)

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), min_size=1, unique_by=lambda x: x[0]))
    def test_remove_elements(self, items):
        ll = LinkedList()
        for k, v in items:
            ll.append(k, v)
            
        # ------------------------------------------------------------------ #
        # Chave que não existe na lista                                      #
        # ------------------------------------------------------------------ #
        non_existent_key = "non_existent_key_prefix_" + "".join(k for k, _ in items)
        self.assertFalse(ll.remove(non_existent_key))
        self.assertEqual(len(ll), len(items))
        
        # ------------------------------------------------------------------ #
        # Remove um elemento arbitrário                                      #
        # ------------------------------------------------------------------ #
        idx_to_remove = len(items) // 2
        key_to_remove = items[idx_to_remove][0]
        
        self.assertTrue(ll.remove(key_to_remove))
        self.assertEqual(len(ll), len(items) - 1)
        
        # ------------------------------------------------------------------ #
        # Garante que não está mais na lista                                 #
        # ------------------------------------------------------------------ #
        keys_left = [node.key for node in ll]
        self.assertNotIn(key_to_remove, keys_left)


# ------------------------------------------------------------------ #
# Testes do HashTable com Hypothesis (Differential Testing)          #
# ------------------------------------------------------------------ #
class TestHypothesisHashTable(unittest.TestCase):

    @given(st.lists(st.one_of(
        st.tuples(st.just("put"), st.text(min_size=1, max_size=10), st.integers()),
        st.tuples(st.just("get"), st.text(min_size=1, max_size=10)),
        st.tuples(st.just("contains"), st.text(min_size=1, max_size=10)),
        st.tuples(st.just("delete"), st.text(min_size=1, max_size=10)),
    ), min_size=0, max_size=50))
    def test_hash_table_differential(self, actions):
        ref_dict = {}
        # ------------------------------------------------------------------ #
        # Capacidade pequena para testar redimensionamentos freqüentes       #
        # ------------------------------------------------------------------ #
        ht = HashTable(capacity=4)
        
        for action in actions:
            op = action[0]
            if op == "put":
                _, key, val = action
                ref_dict[key] = val
                ht.put(key, val)
            elif op == "get":
                _, key = action
                self.assertEqual(ht.get(key), ref_dict.get(key))
            elif op == "contains":
                _, key = action
                self.assertEqual(ht.contains(key), key in ref_dict)
            elif op == "delete":
                _, key = action
                if key in ref_dict:
                    del ref_dict[key]
                    self.assertTrue(ht.delete(key))
                else:
                    self.assertFalse(ht.delete(key))
            
            # ------------------------------------------------------------------ #
            # Valida invariants após a ação                                      #
            # ------------------------------------------------------------------ #
            self.assertEqual(ht.size, len(ref_dict))
            self.assertEqual(sorted(ht.keys()), sorted(list(ref_dict.keys())))
            for k in ref_dict:
                self.assertEqual(ht.get(k), ref_dict[k])


# ------------------------------------------------------------------ #
# Testes do Graph com Hypothesis                                     #
# ------------------------------------------------------------------ #
class TestHypothesisGraph(unittest.TestCase):

    @given(
        st.lists(st.tuples(st.text(min_size=1, max_size=5), st.text(min_size=1, max_size=10), st.lists(st.text(min_size=1, max_size=5)))),
        st.lists(st.tuples(st.integers(), st.integers(), st.floats(min_value=0.0, max_value=1.0)))
    )
    def test_graph_properties(self, vertex_data, edges):
        g = Graph()
        added_vertices = {}
        for vid, sentiment, words in vertex_data:
            if vid not in added_vertices:
                v = Vertex(vid, sentiment, words)
                g.add_vertex(v)
                added_vertices[vid] = v
                
        for vid in added_vertices:
            self.assertEqual(g.get_vertex(vid), added_vertices[vid])
            
        v_keys = list(added_vertices.keys())
        if len(v_keys) >= 2:
            for idx1, idx2, weight in edges:
                id1 = v_keys[idx1 % len(v_keys)]
                id2 = v_keys[idx2 % len(v_keys)]
                if id1 != id2:
                    g.add_edge(id1, id2, weight)
                    
                    v1 = g.get_vertex(id1)
                    v2 = g.get_vertex(id2)
                    
                    # ------------------------------------------------------------------ #
                    # Verifica se a aresta é bidirecional com o peso correto             #
                    # ------------------------------------------------------------------ #
                    found2 = False
                    for node in v1.neighbors:
                        if node.key == id2:
                            self.assertEqual(node.value, weight)
                            found2 = True
                            break
                    self.assertTrue(found2)
                    
                    # ------------------------------------------------------------------ #
                    # Verifica do outro lado                                             #
                    # ------------------------------------------------------------------ #
                    found1 = False
                    for node in v2.neighbors:
                        if node.key == id1:
                            self.assertEqual(node.value, weight)
                            found1 = True
                            break
                    self.assertTrue(found1)

    @given(
        st.lists(st.tuples(st.text(min_size=1, max_size=5), st.text(), st.lists(st.text())), min_size=2, unique_by=lambda x: x[0]),
        st.lists(st.tuples(st.integers(), st.integers(), st.floats(min_value=0.0, max_value=1.0)))
    )
    def test_graph_remove_vertex(self, vertex_data, edges):
        g = Graph()
        added_vertices = {}
        for vid, sentiment, words in vertex_data:
            v = Vertex(vid, sentiment, words)
            g.add_vertex(v)
            added_vertices[vid] = v
            
        v_keys = list(added_vertices.keys())
        for idx1, idx2, weight in edges:
            id1 = v_keys[idx1 % len(v_keys)]
            id2 = v_keys[idx2 % len(v_keys)]
            if id1 != id2:
                g.add_edge(id1, id2, weight)
                
        target_id = v_keys[0]
        neighbors_of_target = [node.key for node in g.get_vertex(target_id).neighbors]
        
        # ------------------------------------------------------------------ #
        # Remove o vértice e valida                                          #
        # ------------------------------------------------------------------ #
        self.assertTrue(g.remove_vertex(target_id))
        self.assertIsNone(g.get_vertex(target_id))
        
        # ------------------------------------------------------------------ #
        # Garante que nenhum vizinho tenha referências para o vértice removido#
        # ------------------------------------------------------------------ #
        for nid in neighbors_of_target:
            nv = g.get_vertex(nid)
            if nv:
                self.assertNotIn(target_id, [node.key for node in nv.neighbors])


# ------------------------------------------------------------------ #
# Testes do Preprocessing com Hypothesis                             #
# ------------------------------------------------------------------ #
class TestHypothesisPreprocessing(unittest.TestCase):

    @given(st.text())
    def test_preprocess_tweet_invariants(self, text):
        words = preprocess_tweet(text)
        self.assertIsInstance(words, list)
        for word in words:
            self.assertIsInstance(word, str)
            self.assertEqual(word, word.lower())
            self.assertNotIn(word, DOMAIN_STOPWORDS)
            self.assertTrue(word.strip() == word)
            self.assertTrue(not any(c.isspace() for c in word))

    @given(st.one_of(st.integers(), st.floats(), st.booleans(), st.none()))
    def test_preprocess_tweet_invalid_inputs(self, invalid_input):
        self.assertEqual(preprocess_tweet(invalid_input), [])


# ------------------------------------------------------------------ #
# Testes do Opinion Mining com Hypothesis                            #
# ------------------------------------------------------------------ #
class TestHypothesisOpinionMining(unittest.TestCase):

    @given(
        st.lists(
            st.fixed_dictionaries({
                'id': st.text(min_size=1, max_size=5),
                'tweet': st.text(min_size=5, max_size=50),
                'sentiment': st.sampled_from(['positive', 'negative', 'neutral'])
            }),
            min_size=3,
            max_size=15,
            unique_by=lambda x: x['id']
        ),
        st.text(min_size=1, max_size=50)
    )
    # ------------------------------------------------------------------ #
    # deadline=None para evitar quebras por lentidão do spaCy            #
    # ------------------------------------------------------------------ #
    @settings(max_examples=15, deadline=None)
    def test_classify_tweet_returns_valid_sentiment(self, tweets_data, query_text):
        graph, inverted_index = build_base_graph(tweets_data, threshold=0.01)
        
        sentiment = classify_tweet(graph, inverted_index, query_text, threshold=0.01)
        self.assertIn(sentiment, ['positive', 'negative', 'neutral'])
        self.assertIsNone(graph.get_vertex("__temp_query_tweet__"))


if __name__ == "__main__":
    unittest.main()
