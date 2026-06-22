# ------------------------------------------------------------------ #
# Classe Node                                                         #
# Um nó em uma lista encadeada simples.                               #
# Armazena uma chave (key), um valor (value) e ponteiro para próximo. #
# ------------------------------------------------------------------ #
class Node:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.next = None


# ------------------------------------------------------------------ #
# Classe LinkedList (Lista Encadeada Simples)                         #
# Possui ponteiro de cauda (tail) para inserções em O(1).            #
# Usada como lista de adjacência dos vértices e nos buckets da hash. #
# ------------------------------------------------------------------ #
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    # ------------------------------------------------------------------ #
    # Insere um novo nó no final da lista em tempo O(1).                 #
    # ------------------------------------------------------------------ #
    def append(self, key, value=None):
        new_node = Node(key, value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    # ------------------------------------------------------------------ #
    # Remove o primeiro nó que contém a chave especificada.              #
    # Retorna True se removido, False caso contrário.                    #
    # ------------------------------------------------------------------ #
    def remove(self, key):
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

    # ------------------------------------------------------------------ #
    # Itera sobre os nós da lista.                                        #
    # ------------------------------------------------------------------ #
    def __iter__(self):
        curr = self.head
        while curr:
            yield curr
            curr = curr.next

    def __len__(self):
        return self.size


# ------------------------------------------------------------------ #
# Classe HashTable (Tabela Hash)                                      #
# Tabela hash customizada com resolução de colisão por encadeamento  #
# e redimensionamento dinâmico quando o fator de carga excede 0.75.  #
# ------------------------------------------------------------------ #
class HashTable:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.size = 0
        self.buckets = [None] * capacity

    # ------------------------------------------------------------------ #
    # Calcula o índice do bucket para uma chave.                         #
    # ------------------------------------------------------------------ #
    def _hash(self, key):
        return abs(hash(key)) % self.capacity

    # ------------------------------------------------------------------ #
    # Insere ou atualiza um par chave-valor.                             #
    # Redimensiona se o fator de carga exceder 0.75.                    #
    # ------------------------------------------------------------------ #
    def put(self, key, value):
        idx = self._hash(key)
        if self.buckets[idx] is None:
            self.buckets[idx] = LinkedList()

        # Verifica se a chave já existe e atualiza
        for node in self.buckets[idx]:
            if node.key == key:
                node.value = value
                return

        self.buckets[idx].append(key, value)
        self.size += 1

        # Redimensiona se fator de carga > 0.75
        if self.size / self.capacity > 0.75:
            self._resize()

    # ------------------------------------------------------------------ #
    # Recupera o valor associado a uma chave.                            #
    # Retorna None se não encontrado.                                    #
    # ------------------------------------------------------------------ #
    def get(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        if bucket is None:
            return None
        for node in bucket:
            if node.key == key:
                return node.value
        return None

    # ------------------------------------------------------------------ #
    # Verifica se uma chave está presente na tabela hash.                #
    # ------------------------------------------------------------------ #
    def contains(self, key):
        return self.get(key) is not None

    # ------------------------------------------------------------------ #
    # Remove um par chave-valor da tabela hash.                          #
    # Retorna True se removido, False caso contrário.                    #
    # ------------------------------------------------------------------ #
    def delete(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        if bucket is None:
            return False
        
        removed = bucket.remove(key)
        if removed:
            self.size -= 1
            # Limpa referência de bucket vazio
            if len(bucket) == 0:
                self.buckets[idx] = None
        return removed

    # ------------------------------------------------------------------ #
    # Dobra a capacidade e rehash de todos os elementos.                 #
    # ------------------------------------------------------------------ #
    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [None] * self.capacity
        self.size = 0  # put incrementará self.size

        for bucket in old_buckets:
            if bucket is not None:
                for node in bucket:
                    self.put(node.key, node.value)

    # ------------------------------------------------------------------ #
    # Retorna uma lista Python com todas as chaves da tabela hash.       #
    # ------------------------------------------------------------------ #
    def keys(self):
        all_keys = []
        for bucket in self.buckets:
            if bucket is not None:
                for node in bucket:
                    all_keys.append(node.key)
        return all_keys


# ------------------------------------------------------------------ #
# Classe Vertex (Vértice)                                             #
# Representa um tweet no Grafo Base.                                  #
# Armazena: id do tweet, sentimento, palavras úteis e vizinhos.      #
# ------------------------------------------------------------------ #
class Vertex:
    def __init__(self, tweet_id, sentiment, useful_words):
        self.id = tweet_id
        self.sentiment = sentiment
        self.useful_words = useful_words
        self.neighbors = LinkedList()


# ------------------------------------------------------------------ #
# Classe Graph (Grafo Não-Direcionado)                                #
# Implementado do zero usando a HashTable customizada para vértices. #
# Suporta adição/remoção de vértices e arestas bidirecionais.        #
# ------------------------------------------------------------------ #
class Graph:
    def __init__(self):
        self.vertices = HashTable(capacity=1000)

    # ------------------------------------------------------------------ #
    # Adiciona um Vértice ao grafo.                                      #
    # ------------------------------------------------------------------ #
    def add_vertex(self, vertex):
        self.vertices.put(vertex.id, vertex)

    # ------------------------------------------------------------------ #
    # Recupera um Vértice pelo seu ID.                                   #
    # ------------------------------------------------------------------ #
    def get_vertex(self, tweet_id):
        return self.vertices.get(tweet_id)

    # ------------------------------------------------------------------ #
    # Adiciona ou atualiza uma aresta não-direcionada entre dois vértices.#
    # A aresta é criada em ambas as direções (bidirecional).             #
    # ------------------------------------------------------------------ #
    def add_edge(self, id1, id2, weight):
        v1 = self.get_vertex(id1)
        v2 = self.get_vertex(id2)
        if v1 and v2:
            self._add_directed_edge(v1, id2, weight)
            self._add_directed_edge(v2, id1, weight)

    # ------------------------------------------------------------------ #
    # Auxiliar interno: adiciona aresta direcionada de um vértice a outro.#
    # Se a aresta já existe, atualiza o peso.                            #
    # ------------------------------------------------------------------ #
    def _add_directed_edge(self, v_from, to_id, weight):
        for node in v_from.neighbors:
            if node.key == to_id:
                node.value = weight
                return
        v_from.neighbors.append(to_id, weight)

    # ------------------------------------------------------------------ #
    # Remove um vértice e todas as arestas correspondentes do grafo.     #
    # Remove referências de todos os vizinhos antes de deletar.          #
    # ------------------------------------------------------------------ #
    def remove_vertex(self, tweet_id):
        v = self.get_vertex(tweet_id)
        if not v:
            return False

        # Remove referência das listas de adjacência de todos os vizinhos
        for node in v.neighbors:
            neighbor_id = node.key
            neighbor_vertex = self.get_vertex(neighbor_id)
            if neighbor_vertex:
                neighbor_vertex.neighbors.remove(tweet_id)

        # Deleta da tabela hash de vértices
        return self.vertices.delete(tweet_id)