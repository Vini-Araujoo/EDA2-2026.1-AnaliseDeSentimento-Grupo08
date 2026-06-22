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

    def remove(self, key):
        """Remove the first node containing the specified key. Returns True if removed, else False."""
        curr = self.head
        prev = None
        while curr:
            if curr.key == key:
                if prev is None:
                    self.head = curr.next
                    if self.head is None:
                        self.tail = None
                else:
                    prev.next = curr.next
                    if curr.next is None:
                        self.tail = prev
                self.size -= 1
                return True
            prev = curr
            curr = curr.next
        return False

    def __iter__(self):
        """Iterate over the nodes of the list."""
        curr = self.head
        while curr:
            yield curr
            curr = curr.next

    def __len__(self):
        return self.size


class HashTable:
    """A custom Hash Table with chaining collision resolution and dynamic resizing."""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.size = 0
        self.buckets = [None] * capacity

    def _hash(self, key):
        """Compute the bucket index for a key."""
        return abs(hash(key)) % self.capacity

    def put(self, key, value):
        """Insert or update a key-value pair. Resizes if load factor exceeds 0.75."""
        idx = self._hash(key)
        if self.buckets[idx] is None:
            self.buckets[idx] = LinkedList()

        # Check if the key already exists and update it
        for node in self.buckets[idx]:
            if node.key == key:
                node.value = value
                return

        self.buckets[idx].append(key, value)
        self.size += 1

        # Resize if load factor is > 0.75
        if self.size / self.capacity > 0.75:
            self._resize()