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
