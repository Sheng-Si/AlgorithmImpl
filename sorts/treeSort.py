"""
基于Binary Search Tree(BST)排序
BST是对于任意的node x，如果node y是node x的左边的节点, 那么Key(y) <= Key(x);
对于任意的node x， 如果node y 是node x的右边的节点， 那么key(y)>=key(x).

Create by kyle 2019.04.30
"""


class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def insert(self, val):
        """插入数据"""
        if self.val:
            if val < self.val:  # 要插入的值小于父节点值,则插入到左节点
                if self.left is None:
                    self.left = Node(val)
                else:
                    self.left.insert(val)
            elif val > self.val:  # 要插入的值大于父节点值,则插入到又节点
                if self.right is None:
                    self.right = Node(val)
                else:
                    self.right.insert(val)
            else:
                self.val = val


class RepeatNode(object):
    def __init__(self, val_list):
        """可重复"""
        self.val_list = val_list
        self.left = None
        self.right = None

    def insert(self, val_list):
        """插入数据"""
        if self.val_list:
            if val_list[0] < self.val_list[0]:  # 要插入的值小于父节点值,则插入到左节点
                if self.left is None:
                    self.left = RepeatNode(val_list)
                else:
                    self.left.insert(val_list)
            elif val_list[0] > self.val_list[0]:  # 要插入的值大于父节点值,则插入到又节点
                if self.right is None:
                    self.right = RepeatNode(val_list)
                else:
                    self.right.insert(val_list)
            else:
                self.val_list += val_list


def in_order(root, res):
    """遍历树并排好序"""
    if root:
        in_order(root.left, res)
        res.append(root.val)
        in_order(root.right, res)


def tree_sort(arr):
    """使用树排序"""
    sort_length = len(arr)
    if sort_length <= 1:
        return arr
    root = Node(arr[0])
    for index in range(1, sort_length):
        root.insert(arr[index])
    res = []
    in_order(root, res)
    return res


def repeat_in_order(root, res):
    """遍历树并排好序,支持重复元素"""
    if root:
        repeat_in_order(root.left, res)
        res += root.val_list
        repeat_in_order(root.right, res)


def repeat_tree_sort(arr):
    """使用树排序,支持重复元素"""
    sort_length = len(arr)
    if sort_length <= 1:
        return arr
    root = RepeatNode([arr[0]])
    for index in range(1, sort_length):
        root.insert([arr[index]])
    res = []
    repeat_in_order(root, res)
    return res


if __name__ == '__main__':
    print(tree_sort([3, 1, 6, 8, 12, -1, 0, 12]))
    print(repeat_tree_sort([3, 1, 6, 8, 12, -1, 0, 12]))
