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

