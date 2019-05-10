# coding=utf-8
"""
给定一个待选数组和一个目标数,在待选数组中寻找哪两个数之和等于目标数

Create by kyle 2019.05.08
"""


def two_sum(arr, target):
    tmp = {}
    for index, n in enumerate(arr):
        diff = target - n
        cache = tmp.get(diff)
        if cache:
            return [n, arr[cache]]
        else:
            tmp[n] = index
    return []


print(two_sum([1, 2, 3, 4, 1, 1, 4, 5, 6, 1, 23, 5123, 123, 15, 1], 100))
