class Node:
    """A node in a singly linked list."""
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.next = None