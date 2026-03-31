class UnionFind:
    def __init__(self):
        self.parent = {}

    # def find(self, x):
    #     if x not in self.parent:
    #         self.parent[x] = x
    #         return x
    #     if self.parent[x] != x:
    #         self.parent[x] = self.find(self.parent[x])
    #     return self.parent[x
    def find(self, x):
        # Find the root in the first pass
        root = x
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[root] != root:
            root = self.parent[root]
        
        # Path compression in the second pass
        curr = x
        while self.parent[curr] != root:
            next_node = self.parent[curr]
            self.parent[curr] = root
            curr = next_node
            
        return root

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y

    def clear(self):
        self.parent.clear()
