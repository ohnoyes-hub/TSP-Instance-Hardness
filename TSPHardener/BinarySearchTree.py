class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def search(root, target):
    if root is None:
        return False
    if root.value > target:
        return search(root.right, target)
    elif target < root.value:
        return search(root.left, target)
    else:
        return True
    
def insert(root, value):
    if root is None:
        return TreeNode(value)
    if root.value > value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root

def findMinValueNode(node):
    current = node
    while current.left is not None:
        current = current.left
    return current

def remove(root, value):
    if not root:
        return None
    if value > root.value:
        root.right = remove(root.right, value)
    elif value < root.value:
        root.left = remove(root.left, value)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        minValue = findMinValueNode(root.right)
        root.value = minValue.value
        root.right = remove(root.right, minValue.value)
    return root