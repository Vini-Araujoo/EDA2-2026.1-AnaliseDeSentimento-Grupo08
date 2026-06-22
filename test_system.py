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