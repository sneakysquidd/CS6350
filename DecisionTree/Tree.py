class Node(object):
    def __init__(self, data, depth):
        self.data = data
        self.children = []
        self.depth = depth

        def add_child(self, obj):
            self.children.append(obj)

