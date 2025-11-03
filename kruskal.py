class Grafo:
    def __init__(self, vertices):
        self.V = vertices
        self.adj = {i: [] for i in range(vertices)}
        self.arestas = []

    def adicionar_aresta(self, u, v, peso):
        self.adj[u].append((v, peso))
        self.adj[v].append((u, peso))  
        self.arestas.append((peso, u, v))

    
    def find(self, pai, i):
        if pai[i] != i:
            pai[i] = self.find(pai, pai[i])
        return pai[i]

    def union(self, pai, rank, x, y):
        raiz_x = self.find(pai, x)
        raiz_y = self.find(pai, y)

        if rank[raiz_x] < rank[raiz_y]:
            pai[raiz_x] = raiz_y
        elif rank[raiz_x] > rank[raiz_y]:
            pai[raiz_y] = raiz_x
        else:
            pai[raiz_y] = raiz_x
            rank[raiz_x] += 1

    
    def kruskal(self):
        resultado = []  
        i, e = 0, 0  

        # Ordena todas as arestas pelo peso
        self.arestas.sort(key=lambda x: x[0])

        pai = []
        rank = []

        for no in range(self.V):
            pai.append(no)
            rank.append(0)

        # Escolhe as menores arestas 
        while e < self.V - 1 and i < len(self.arestas):
            peso, u, v = self.arestas[i]
            i += 1

            x = self.find(pai, u)
            y = self.find(pai, v)

            # Se não formam ciclo, adiciona
            if x != y:
                e += 1
                resultado.append((u, v, peso))
                self.union(pai, rank, x, y)

        
        return resultado


if __name__ == "__main__":
    g = Grafo(5)
    g.adicionar_aresta(0, 1, 10)
    g.adicionar_aresta(0, 2, 6)
    g.adicionar_aresta(0, 3, 5)
    g.adicionar_aresta(1, 3, 15)
    g.adicionar_aresta(2, 3, 4)

    agm = g.kruskal()

    print("Arestas da Árvore Geradora Mínima:")
    for u, v, peso in agm:
        print(f"{u} -- {v}  peso = {peso}")
