import cv2
import numpy as np
import random




class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)
        self.int_diff = np.zeros(n, dtype=float)  

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, u, v, w, k):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False

        # Critério de fusão adaptativo
        t_u = self.int_diff[ru] + k / self.size[ru]
        t_v = self.int_diff[rv] + k / self.size[rv]
        if w <= min(t_u, t_v):
            # União por tamanho
            if self.size[ru] < self.size[rv]:
                ru, rv = rv, ru
            self.parent[rv] = ru
            self.size[ru] += self.size[rv]
            self.int_diff[ru] = max(self.int_diff[ru], self.int_diff[rv], w)
            return True
        return False


#  (lista dinâmica)

def construir_grafo(img):
    h, w = img.shape[:2]
    edges = []
    idx = lambda x, y: x * w + y

    # Conectividade 4-vizinhança
    for x in range(h):
        for y in range(w):
            if x + 1 < h:
                peso = np.sum(np.abs(img[x, y] - img[x + 1, y]))  # Manhattan RGB
                edges.append((idx(x, y), idx(x + 1, y), peso))
            if y + 1 < w:
                peso = np.sum(np.abs(img[x, y] - img[x, y + 1]))
                edges.append((idx(x, y), idx(x, y + 1), peso))
    return edges, h * w, h, w





def segmentar_mst(img, k=500):
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    edges, n, h, w = construir_grafo(img)
    edges.sort(key=lambda e: e[2])

    uf = UnionFind(n)
    for u, v, peso in edges:
        uf.union(u, v, peso, k)

    # Rótulos finais
    labels = np.zeros(n, dtype=int)
    label_map = {}
    next_label = 0

    for i in range(n):
        r = uf.find(i)
        if r not in label_map:
            label_map[r] = next_label
            next_label += 1
        labels[i] = label_map[r]

    labels = labels.reshape((h, w))
    return labels, next_label



# Visualização colorida das regiões

def colorir_segmentos(labels, num_segments, img_original):
    h, w = labels.shape
    out = np.zeros_like(img_original, dtype=np.uint8)

    # Calcula cor média por região
    for i in range(num_segments):
        mask = (labels == i)
        if np.any(mask):
            mean_color = np.mean(img_original[mask], axis=0)
            out[mask] = mean_color
    return out




if __name__ == "__main__":
    caminho = "teste.jpg"  
    img = cv2.imread(caminho)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    for k in [800, 8000, 20000]: 
        labels, nseg = segmentar_mst(img, k=k)
        colorida = colorir_segmentos(labels, nseg, img)
        cv2.imwrite(f"segmentacao_k_{k}.png", colorida)
        print(f"Segmentação concluída para k={k}, regiões detectadas: {nseg}")
