import unittest
from data_structures import Node, LinkedList, HashTable, Vertex, Graph
from preprocessing import preprocess_tweet
from opinion_mining import build_base_graph, classify_tweet

class TestDataStructures(unittest.TestCase):
    def test_linked_list_append_and_remove(self):
        ll = LinkedList()
        self.assertEqual(len(ll), 0)
        
        ll.append("A", 10)
        ll.append("B", 20)
        ll.append("C", 30)
        self.assertEqual(len(ll), 3)
        
        # Check iteration
        nodes = list(ll)
        self.assertEqual(nodes[0].key, "A")
        self.assertEqual(nodes[0].value, 10)
        self.assertEqual(nodes[1].key, "B")
        self.assertEqual(nodes[2].key, "C")
        
        # Check tail
        self.assertEqual(ll.tail.key, "C")
        
        # Remove middle
        self.assertTrue(ll.remove("B"))
        self.assertEqual(len(ll), 2)
        nodes = list(ll)
        self.assertEqual(nodes[0].key, "A")
        self.assertEqual(nodes[1].key, "C")
        
        # Remove head
        self.assertTrue(ll.remove("A"))
        self.assertEqual(len(ll), 1)
        self.assertEqual(ll.head.key, "C")
        self.assertEqual(ll.tail.key, "C")
        
        # Remove tail
        self.assertTrue(ll.remove("C"))
        self.assertEqual(len(ll), 0)
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)

    def test_hash_table_operations(self):
        ht = HashTable(capacity=4)  # Small capacity to trigger resizing
        ht.put("apple", 1)
        ht.put("banana", 2)
        ht.put("cherry", 3)
        
        self.assertEqual(ht.get("apple"), 1)
        self.assertEqual(ht.get("banana"), 2)
        self.assertEqual(ht.get("cherry"), 3)
        self.assertTrue(ht.contains("banana"))
        self.assertFalse(ht.contains("dragonfruit"))
        
        # Resizing check
        ht.put("date", 4)  # 4 elements in capacity 4 -> load factor 1.0 > 0.75 -> resizing triggers
        self.assertEqual(ht.capacity, 8)
        self.assertEqual(ht.get("apple"), 1)
        self.assertEqual(ht.get("date"), 4)
        
        # Overwrite value
        ht.put("apple", 10)
        self.assertEqual(ht.get("apple"), 10)
        
        # Delete value
        self.assertTrue(ht.delete("banana"))
        self.assertIsNone(ht.get("banana"))
        self.assertFalse(ht.contains("banana"))
        self.assertEqual(ht.size, 3)
        
        # Delete non-existent
        self.assertFalse(ht.delete("banana"))

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
        
        # Verify edge exists bidirectionally
        # Vertex 1 neighbor should be Vertex 2 with weight 5
        v1_neighbors = list(v1.neighbors)
        self.assertEqual(len(v1_neighbors), 1)
        self.assertEqual(v1_neighbors[0].key, "2")
        self.assertEqual(v1_neighbors[0].value, 5)
        
        # Vertex 2 should have neighbor 1 (weight 5) and 3 (weight 3)
        v2_neighbors = list(v2.neighbors)
        self.assertEqual(len(v2_neighbors), 2)
        v2_neighbor_keys = [node.key for node in v2_neighbors]
        self.assertIn("1", v2_neighbor_keys)
        self.assertIn("3", v2_neighbor_keys)
        
        # Remove Vertex 2
        g.remove_vertex("2")
        self.assertIsNone(g.get_vertex("2"))
        
        # Vertex 1 should no longer list Vertex 2 as neighbor
        self.assertEqual(len(list(v1.neighbors)), 0)
        # Vertex 3 should no longer list Vertex 2 as neighbor
        self.assertEqual(len(list(v3.neighbors)), 0)

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_tweet(self):
        tweet = "The players are playing beautifully! It is a great victory, but I'm tired."
        words = preprocess_tweet(tweet)
        
        # "playing" -> "play"
        # "victory" -> "victory"
        # Stopwords "the", "are", "beautifully" (beautifully might be stopword or not, but let's check basic ones)
        # "is", "a", "but", "I'm", "tired"
        # Punctuation "!." removed
        self.assertIn("play", words)
        self.assertIn("victory", words)
        self.assertNotIn("are", words)
        self.assertNotIn("is", words)
        
        # Confirm lowercase conversion
        for w in words:
            self.assertEqual(w, w.lower())
