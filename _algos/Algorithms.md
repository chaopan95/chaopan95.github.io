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


## 搜索
### 二分
### BFS
### DFS (回溯 Backtracking)
#### 组合
##### Leetcode 39. 组合总和
{% highlight C++ %}
/*
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出
candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

示例 1：
输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]

示例 2：
输入：candidates = [2,3,5], target = 8,
所求解集为：
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
*/
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> res;
    vector<int> arr;
    int n = int(candidates.size());
    if (n == 0) { return res; }
    DFS(res, arr, candidates, n, target, 0);
    return res;
}
void DFS(vector<vector<int>> &res, vector<int> &arr,
          vector<int> candidates, int n, int target,
          int idx)
{
    if (target == 0)
    {
        res.push_back(arr);
        return;
    }
    for (int i = idx; i < n; i++)
    {
        if (target-candidates[i] >= 0)
        {
            arr.push_back(candidates[i]);
            DFS(res, arr, candidates, n, target-candidates[i], i);
            arr.resize(arr.size()-1);
        }
    }
}
{% endhighlight %}

##### Leetcode 40. 组合总和 II
{% highlight C++ %}
/*
给定一个数组 candidates 和一个目标数 target ，找出
candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明：
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 

示例 1:
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

示例 2:
输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]
*/
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>> res;
    vector<int> arr;
    int n = int(candidates.size());
    if (n == 0) { return res; }
    map<int, int> dict;
    map<int, int>::iterator iter;
    for (int num: candidates)
    {
        iter = dict.find(num);
        if (iter == dict.end()) { dict[num] = 1; }
        else { dict[num]++; }
    }
    vector<pair<int, int>> nums;
    int size = 0;
    for (iter = dict.begin(); iter != dict.end(); iter++)
    {
        nums.emplace_back(iter->first, iter->second);
        size++;
    }
    DFS(res, arr, candidates, target, 0, size, nums);
    return res;
}
void DFS(vector<vector<int>> &res, vector<int> &arr,
          vector<int> candidates, int target, int idx,
          int size, vector<pair<int, int>> nums)
{
    if (target <= 0 || idx >= size)
    {
        if (target == 0) { res.push_back(arr); }
        return;
    }
    for (int j = 1; j <= nums[idx].second; j++)
    {
        for (int k = 0; k < j; k++) { arr.push_back(nums[idx].first); }
        DFS(res, arr, candidates, target-j*nums[idx].first, idx+1, size,
            nums);
        for (int k = 0; k < j; k++) { arr.pop_back(); }
    }
    DFS(res, arr, candidates, target, idx+1, size, nums);
}
{% endhighlight %}


### A*
{% highlight C++ %}

{% endhighlight %}


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
### 求一个数组的元素和
{% highlight C++ %}
int sum(int A[], int lo, int hi) {
    if (lo == hi) { return A[lo]; }
    int mi = (lo + hi) >> 1;
    return sum(A, lo, mi) + sum(A, mi+1, hi);
}
{% endhighlight %}
<p align="justify">
时间复杂度分析
$$
\begin{aligned}
T(n) &= 2T(\frac{n}{2}) \\
&= 2 \cdot 2 T(\frac{n}{4}) \\
&= 2^{\log_{2}n} \cdot T(1) \\
&= n
\end{aligned}
$$
</p>

## 减治 Decrease and Conquer
### 将一个数组反转
{% highlight C++ %}
void reverse(int *A, int lo, int hi)
{
    if (lo < hi) {
        swap(A[lo], A[hi]);
        reverse(A, lo+1, hi-1);
    }
}
{% endhighlight %}


## 数学
### 最大公约数
#### 两数最大公约数
{% highlight C++ %}
/*
Brutal force: enumerate all numbers from 1 to min(a, b), save the greatest.
*/
int GCD_enumerate(int a, int b)
{
    int c = (a < b ? a : b);
    int gcd = 1;
    for (int i = 1; i <= c; i++)
    {
        if (a % i == 0 && b % i == 0) { gcd = i; }
    }
    return gcd;
}

int GCD_Euclidean_Algorithm_divide(int a, int b)
{
    while (a * b)
    {
        if (a < b) { b %= a; }
        else { a %= b; }
    }
    return (a > b ? a : b );
}

int GCD_Euclidean_Algorithm_subtract(int a, int b)
{
    while (a * b)
    {
        if (a < b) { b -= a; }
        else { a -= b; }
    }
    return (a > b ? a : b);
}
{% endhighlight %}

#### 数组的最大公约数
{% highlight C++ %}
int GCD_Array(vector<int> arr)
{
    int n = int(arr.size());
    if (n == 1) { return arr[0]; }
    int gcd = GCD_Euclidean_Algorithm_divide(arr[0], arr[1]);
    for (int i = 2; i < n; i++)
    {
        gcd = GCD_Euclidean_Algorithm_divide(gcd, arr[i]);
    }
    return gcd;
}
{% endhighlight %}

#### 最大公倍数
<p align="justify">
对于两数a和b的最大公倍数 $\text{LCM} = \frac{a * b}{\text{GCD}(a, b)}$
</p>
{% highlight C++ %}
int LCM(int a, int b) {
    int gcd = GCD_Euclidean_Algorithm_divide(a, b);
    return (a * b) / gcd;
}
{% endhighlight %}

## 数据结构
### 数组
### 链表
#### 单链表
##### Leetcode 206. 反转链表
{% highlight C++ %}
/*
from 1 -> 2 -> 3 -> 4 -> 5
to   5 -> 4 -> 3 -> 2 -> 1
*/
// 迭代
ListNode* reverseList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) { return head; }
    ListNode *p1 = head, *p2 = head->next, *p3 = head->next->next;
    while (p3 != nullptr)
    {
        p2->next = p1;
        p1 = p2;
        p2 = p3;
        p3 = p3->next;
    }
    p2->next = p1;
    head->next = nullptr;
    return p2;
}

// 递归
ListNode* reverseList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) { return head; }
    ListNode *ans = reverse(head, head->next);
    head->next = nullptr;
    return ans;
}
ListNode *reverse(ListNode *p1, ListNode *p2)
{
    if (p2->next != nullptr)
    {
        ListNode *p3 = p2->next;
        p2->next = p1;
        return reverse(p2, p3);
    }
    p2->next = p1;
    return p2;
}
{% endhighlight %}

#### 双链表
##### Leetcode 0146. LRU 缓存机制
{% highlight C++ %}
/*
运用你所掌握的数据结构，设计和实现一个LRU(最近最少使用)缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键
字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前
删除最久未使用的数据值，从而为新的数据值留出空间。

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
*/
struct BiNode {
    int key, val;
    BiNode *prec, *next;
    BiNode(int k, int v): key(k), val(v), prec(nullptr), next(nullptr) {}
};


class LRUCache {
    BiNode *head;
    BiNode *tail;
    int cap;
    unordered_map<int, BiNode*> hash;
public:
    LRUCache(int capacity) {
        cap = capacity;
        head = new BiNode(0, 0);
        tail = new BiNode(0, 0);
        head->next = tail;
        tail->prec = head;
    }
    ~LRUCache () {
        BiNode *curNode = head->next;
        while (curNode != nullptr) {
            delete curNode->prec;
            curNode = curNode->next;
        }
        delete tail;
    }
    
    int get(int key) {
        if (hash.find(key) != hash.end()) {
            BiNode *node = hash[key];
            int value = node->val;
            isolateNode(node);
            setFirstNode(node);
            return value;
        }
        else { return -1; }
    }
    
    void put(int key, int value) {
        BiNode *node;
        if (hash.find(key) == hash.end()) {
            node = new BiNode(key, value);
            cap--;
        }
        else {
            node = hash[key];
            node->val = value;
            isolateNode(node);
        }
        setFirstNode(node);
        hash[key] = node;
        if (cap < 0) {
            hash.erase(tail->prec->key);
            deleteNode(tail->prec);
            cap++;
        }
    }
    
    void setFirstNode(BiNode *node) {
        head->next->prec = node;
        node->next = head->next;
        node->prec = head;
        head->next = node;
    }
    
    void isolateNode(BiNode *node) {
        node->prec->next = node->next;
        node->next->prec = node->prec;
    }
    
    void deleteNode(BiNode *node) {
        isolateNode(node);
        delete node;
    }
};
/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
{% endhighlight %}

### 栈
#### 汉诺塔
{% highlight C++ %}
int hanoi(int n, char x, char y, char z) {
    if (n == 1) {
        printf("%d: %c -> %c\n", n, x, z);
        return 1;
    }
    else {
        // 1至n-1移到y柱，借助z
        int res = hanoi(n-1, x, z, y);
        // n移到z
        printf("%d: %c -> %c\n", n, x, z);
        // 1至n-1移到z柱，借助x
        res += hanoi(n-1, y, x, z);
        return 1+res;
    }
}
{% endhighlight %}

#### 进制转换
{% highlight C++ %}

{% endhighlight %}

### 队列
### 树
#### 二叉树
##### 前序遍历
##### 中序遍历
##### 后序遍历
##### 层序遍历
##### 线索树
#### 并查集
#### 线段树
#### 字典树
##### Leetcode 208. 实现 Trie (前缀树)
{% highlight C++ %}
struct TrieNode {
    vector<char> val;
    vector<TrieNode*> next;
    bool isEnd;
    TrieNode() {
        val.resize(26, ' ');
        next.resize(26, nullptr);
        isEnd = false;
    }
    TrieNode(char x) {
        val.resize(26, ' ');
        val[x - 'a'] = x;
        next.resize(26, nullptr);
        isEnd = false;
    }
};

class Trie {
    TrieNode *root;
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode *curNode = root;
        for (const char &ch : word) {
            int idx = ch - 'a';
            if (curNode->next[idx] == nullptr) {
                curNode->val[idx] = ch;
                curNode->next[idx] = new TrieNode();
            }
            curNode = curNode->next[idx];
        }
        curNode->isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode *curNode = root;
        for (const char &ch : word) {
            int idx = ch - 'a';
            if (curNode->val[idx] != ch) { return false; }
            curNode = curNode->next[idx];
        }
        return curNode->isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode *curNode = root;
        for (const char &ch : prefix) {
            int idx = ch - 'a';
            if (curNode->val[idx] != ch) { return false; }
            curNode = curNode->next[idx];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
{% endhighlight %}

### 图
### 堆
### 字符串
#### KMP
##### Leetcode 28. 实现 strStr()
{% highlight C++ %}
/*
示例 1：
输入：haystack = "hello", needle = "ll"
输出：2

示例 2：
输入：haystack = "aaaaa", needle = "bba"
输出：-1

示例 3：
输入：haystack = "", needle = ""
输出：0
*/
class Solution {
public:
    int strStr(string haystack, string needle) {
        string T = haystack, P = needle;
        int nT = (int)T.length(), nP = (int)P.length();
        if (nP == 0) { return 0; }
        if (nT == 0) { return -1; }
        int i = 0, j = 0, k = 0, *next = new int [nP]{};
        int t = next[0] = -1;
        while (k < nP - 1) {
            if (t < 0 || P[k] == P[t]) { next[++k] = ++t; }
            else { t = next[t]; }
        }
        while (i < nT && j < nP) {
            if (j < 0 || T[i] == P[j]) { i++; j++; }
            else { j = next[j]; }
        }
        delete []next;
        if (i - j > nT - nP) { return -1; }
        return i - j;
    }
};
{% endhighlight %}

#### Boyer-Moore
#### Rabin-Karp
{% highlight C++ %}

{% endhighlight %}