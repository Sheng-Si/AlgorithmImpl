# coding=utf-8
"""
Median of Two Sorted Arrays

Create by kyle 2019.05.08
"""

from __future__ import division


def find_median_sorted_arrays(nums1, nums2):
    a, b = sorted((nums1, nums2), key=len)
    m, n = len(a), len(b)
    after = (m + n - 1) // 2
    lo, hi = 0, m
    while lo < hi:
        i = (lo + hi) // 2
        if after - i - 1 < 0 or a[i] >= b[after - i - 1]:
            hi = i
        else:
            lo = i + 1
    i = lo
    nextfew = sorted(a[i:i + 2] + b[after - i:after - i + 2])
    return (nextfew[0] + nextfew[1 - (m + n) % 2]) / 2


print(find_median_sorted_arrays([1], [0]))
