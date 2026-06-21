class Node:
    """A node in a singly linked list."""
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.next = None

class LinkedList:
    """A singly linked list with a tail pointer for O(1) appends."""
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, key, value=None):
        """Append a new node to the end of the list in O(1) time."""
        new_node = Node(key, value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1