---
layout: page
title:  "Algorithms"
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


{:toc}

* 
{:toc}



<style>
table {
  border-collapse: collapse;
  border: 1px solid black;
  margin: 0 auto;
} 

th,td {
  border: 1px solid black;
  text-align: center;
  padding: 20px;
}

table.a {
  table-layout: auto;
  width: 180px;  
}

table.b {
  table-layout: fixed;
  width: 600px;  
}

table.c {
  table-layout: auto;
  width: 100%;  
}

table.d {
  table-layout: fixed;
  width: 100%;  
}
</style>


## 回溯 Backtracking
<p align="justify">
<a href="https://chaopan95.github.io/algos/Leetcode/#0039-combination-sum"> Combination Sum</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0040-combination-sum-ii"> Combination Sum II</a><br>
</p>


## 动态规划 Dynamic Programming
<p align="justify">
<a href="https://chaopan95.github.io/algos/Leetcode/#0005-longest-palindromic-substring"> Longest Palindromic Substring</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0010-regular-expression-matching"> Regular Expression Matching</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0032-longest-valid-parentheses"> Longest Valid Parentheses</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0042-trapping-rain-water"> Trapping Rain Water</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0044-wildcard-matching"> Wildcard Matching</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0053-maximum-subarray"> Maximum Subarray</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0062-unique-paths"> Unique Paths</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0063-unique-paths-ii"> Unique Paths II</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0064-minimum-path-sum"> Minimum Path Sum</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0072-edit-distance"> Edit Distance</a><br>
<a href="https://chaopan95.github.io/algos/Leetcode/#0087-scramble-string"> Scramble String</a><br>
</p>

### 序列型DP
#### Leetcode 740. 删除并获得点数
{% highlight C++ %}
/*
给你一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。
之后，你必须删除每个等于 nums[i] - 1 或 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

示例 1：
输入：nums = [3,4,2]
输出：6
解释：
删除 4 获得 4 个点数，因此 3 也被删除。
之后，删除 2 获得 2 个点数。总共获得 6 个点数。

示例 2：
输入：nums = [2,2,3,3,3,4]
输出：9
解释：
删除 3 获得 3 个点数，接着要删除两个 2 和 4 。
之后，再次删除 3 获得 3 个点数，再次删除 3 获得 3 个点数。
总共获得 9 个点数。
*/
class Solution {
public:
    int deleteAndEarn(vector<int>& nums) {
        if (nums.empty()) { return 0; }
        map<int, int> cnt;
        for (int num : nums) { cnt[num]++; }
        map<int, int>::iterator iter = ++cnt.begin();
        int lastN = cnt.begin()->first, lastC = cnt.begin()->second;
        int a = 0, b = lastN * lastC;
        for (; iter != cnt.end(); iter++) {
            int na = max(a, b), nb = iter->first * iter->second;
            if (lastN + 1 == iter->first) {
                nb += a;
            }
            else {
                nb += max(a, b);
            }
            a = na;
            b = nb;
            lastN = iter->first;
        }
        return max(a, b);
    }
};
{% endhighlight %}

## 分治 Divider and Conquer


## Data Structures
### Vector
### List
### Stack
### Queue
### Tree
### Graph
### Heap
### String
{% highlight C++ %}

{% endhighlight %}