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
#### 牛客题霸-算法篇 105. 二分查找-II
{% highlight C++ %}
/*
请实现有重复数字的升序数组的二分查找
给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1

输入
[1,2,4,4,5],4
返回值
2
说明
从左到右，查找到第1个为4的，下标为2，返回2 
*/
/**
 * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
 * 如果目标值存在返回下标，否则返回 -1
 * @param nums int整型vector
 * @param target int整型
 * @return int整型
 */
int search(vector<int>& nums, int target) {
    // write code here
    int n = (int)nums.size();
    if (n == 0) { return -1; }
    int l = 0, r = n - 1;
    while (l < r) {
        int m = (l + r) >> 1;
        if (nums[m] < target) { l = m + 1; }
        else { r = m; }
    }
    return nums[l] == target ? l : -1;
}
{% endhighlight %}

#### Leetcode 33. 搜索旋转排序数组
{% highlight C++ %}
/*
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
*/
int search(vector<int>& nums, int target) {
    int n = int(nums.size());
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        if (nums[m] == target) { return m; }
        if (nums[l] <= nums[r]) {
            if (nums[m] < target) { l = m + 1; }
            else if (nums[m] > target) { r = m - 1; }
        }
        else {
            if (target >= nums[l] && nums[m] >= nums[l]) {
                if (nums[m] > target) { r = m - 1; }
                else { l = m + 1; }
            }
            else if (target >= nums[l] && nums[m] <= nums[r]) { r = m - 1; }
            else if (target <= nums[r] && nums[m] <= nums[r])
            {
                if (nums[m] > target) { r = m - 1; }
                else { l = m + 1; }
            }
            else if (target <= nums[r] && nums[m] >= nums[l]) { l = m + 1; }
            else {
                if (nums[m] < target) { r = m - 1; }
                else { l = m + 1; }
            }
        }
    }
    return -1;
}
{% endhighlight %}

#### Leetcode 33. 搜索旋转排序数组II
{% highlight C++ %}
/*
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
*/
bool search(vector<int>& nums, int target) {
    int n = int(nums.size());
    if (n == 0) { return false; }
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        if (nums[m] == target) { return true; }
        if (nums[l] < nums[r]) {
            if (nums[m] > target) { r = m - 1; }
            else { l = m + 1; }
        }
        else {
            if (nums[m] == nums[l] && nums[m] == nums[r]) { l++; r--; }
            else if ((target >= nums[l] && nums[m] >= nums[l]) ||
                (target <= nums[r] && nums[m] <= nums[r]))
            {
                if (nums[m] > target) { r = m - 1; }
                else { l = m + 1; }
            }
            else if (target >= nums[l] && nums[m] <= nums[r]) { r = m - 1; }
            else if (target <= nums[r] && nums[m] >= nums[l]) { l = m + 1; }
            else
            {
                if (nums[m] > target) { l = m + 1; }
                else { r = m - 1; }
            }
        }
    }
    return false;
}
{% endhighlight %}

#### Leetcode 74. 搜索二维矩阵
{% highlight C++ %}
/*
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。
1   3   5   7
10  11  16  20
23  30  34  60
*/
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int nRow = int(matrix.size());
    if (nRow == 0) { return false; }
    int nCol = int(matrix[0].size());
    if (nCol == 0) { return false; }
    int l = 0, r = nRow * nCol - 1;
    while (l <= r) {
        int m = (l + r) >> 1;
        int i = m / nCol, j = m % nCol;
        if (matrix[i][j] < target) { l = m + 1; }
        else if (matrix[i][j] > target) { r = m - 1; }
        else { return true; }
    }
    return false;
}
{% endhighlight %}

#### Leetcode 1011. 在 D 天内送达包裹的能力
{% highlight C++ %}
/*
传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。传送带上的第 i 个
包裹的重量为 weights[i]。每一天，我们都会按给出重量的顺序往传送带上装载
包裹。我们装载的重量不会超过船的最大运载重量。返回能在 D 天内将传送带上的
所有包裹送达的船的最低运载能力。

示例 1：
输入：weights = [1,2,3,4,5,6,7,8,9,10], D = 5
输出：15
解释：
船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
第 1 天：1, 2, 3, 4, 5
第 2 天：6, 7
第 3 天：8
第 4 天：9
第 5 天：10

请注意，货物必须按照给定的顺序装运，因此使用载重能力为 14 的船舶并将包装
分成 (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) 是不允许的。 
*/
class Solution {
public:
    int shipWithinDays(vector<int>& weights, int D) {
        int l = 0, r = 0;
        for (int weight : weights) {
            if (l < weight) { l = weight; }
            r += weight;
        }
        while (l < r) {
            int m = (l + r) >> 1;
            int days = 1, weightSum = 0;
            for (int weight : weights) {
                if (weightSum + weight > m) {
                    days++;
                    weightSum = 0;
                }
                weightSum += weight;
            }
            if (days > D) { l = m + 1; }
            else { r = m; }
        }
        return l;
    }
};
{% endhighlight %}

#### Leetcode 1482. 制作 m 束花所需的最少天数
{% highlight C++ %}
/*
给你一个整数数组 bloomDay，以及两个整数 m 和 k 。
现需要制作 m 束花。制作花束时，需要使用花园中 相邻的 k 朵花 。
花园中有 n 朵花，第 i 朵花会在 bloomDay[i] 时盛开，恰好 可
以用于一束 花中。
请你返回从花园中摘 m 束花需要等待的最少的天数。如果不能摘到 m 束花则返回 -1 。

示例 1：
输入：bloomDay = [1,10,3,10,2], m = 3, k = 1
输出：3
解释：让我们一起观察这三天的花开过程，x 表示花开，而 _ 表示花还未开。
现在需要制作 3 束花，每束只需要 1 朵。
1 天后：[x, _, _, _, _]   // 只能制作 1 束花
2 天后：[x, _, _, _, x]   // 只能制作 2 束花
3 天后：[x, _, x, _, x]   // 可以制作 3 束花，答案为 3

示例 2：
输入：bloomDay = [1,10,3,10,2], m = 3, k = 2
输出：-1
解释：要制作 3 束花，每束需要 2 朵花，也就是一共需要 6 朵花。而花园中只有 5
朵花，无法满足制作要求，返回 -1 。
*/
class Solution {
public:
    int minDays(vector<int>& bloomDay, int m, int k) {
        int n = (int)bloomDay.size();
        if (n < m * k) { return -1; }
        int minDay = INT_MAX, maxDay = 0;
        for (int day : bloomDay) {
            minDay = min(minDay, day);
            maxDay = max(maxDay, day);
        }
        while (minDay < maxDay) {
            int midDay = (minDay + maxDay) >> 1;
            if (canMake(bloomDay, m, k, midDay)) {
                maxDay = midDay;
            }
            else {
                minDay = midDay + 1;
            }
        }
        return maxDay;
    }
    bool canMake (vector<int> &bloomDay, int m, int k, int days) {
        int bouquets = 0, flowers = 0;
        for (int bd : bloomDay) {
            if (bd <= days) {
                flowers++;
                if (flowers == k) {
                    bouquets++;
                    flowers = 0;
                }
            }
            else {
                flowers = 0;
            }
        }
        return bouquets >= m;
    }
};
{% endhighlight %}

#### LCP 12. 小张刷题计划
{% highlight C++ %}
/*
为了提高自己的代码能力，小张制定了 LeetCode 刷题计划，他选中了 LeetCode 题库中的 n 
道题，编号从 0 到 n-1，并计划在 m 天内按照题目编号顺序刷完所有的题目（注意，小张不能
用多天完成同一题）。
在小张刷题计划中，小张需要用 time[i] 的时间完成编号 i 的题目。此外，小张还可以使用场
外求助功能，通过询问他的好朋友小杨题目的解法，可以省去该题的做题时间。为了防止“小张刷题
计划”变成“小杨刷题计划”，小张每天最多使用一次求助。
我们定义 m 天中做题时间最多的一天耗时为 T（小杨完成的题目不计入做题总时间）。请你帮小
张求出最小的 T是多少。

示例 1：
输入：time = [1,2,3,3], m = 2
输出：3
解释：第一天小张完成前三题，其中第三题找小杨帮忙；第二天完成第四题，并且找小杨帮忙。这样
做题时间最多的一天花费了 3 的时间，并且这个值是最小的。

示例 2：
输入：time = [999,999,999], m = 4
输出：0
解释：在前三天中，小张每天求助小杨一次，这样他可以在三天内完成所有的题目并不花任何时间。
*/
int minTime(vector<int>& time, int m) {
    int n = int(time.size()), tot = 0, minT = INT_MAX;
    if (n == 0 || n <= m) { return 0; }
    for (int t : time) {
        tot += t;
        minT = min(minT, t);
    }
    int l = minT, r = tot;
    while (l < r) {
        int mid = (l + r) >> 1;
        int days = 1, curSum = 0, maxT = 0;
        for (int t : time) {
            if (t > maxT) {
                curSum += maxT;
                maxT = t;
            }
            else {
                curSum += t;
            }
            if (curSum > mid) {
                days++;
                curSum = 0;
                maxT = t;
            }
        }
        if (days > m) {
            l = mid + 1;
        }
        else {
            r = mid;
        }
    }
    return r;
}
{% endhighlight %}

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

### Leetcode 240. 搜索二维矩阵 II
{% highlight C++ %}
/*
1   4   7
2   5   8
3   6   9
find(5) = true
*/
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int nRow = (int)matrix.size();
        if (nRow == 0) { return false; }
        int nCol = (int)matrix[0].size();
        if (nCol == 0) { return false; }
        int row = nRow - 1, col = 0;
        while (row >= 0 && col < nCol) {
            if (matrix[row][col] == target) { return true; }
            else if (matrix[row][col] > target) { row--; }
            else { col++; }
        }
        return false;
    }
};
{% endhighlight %}

## 双指针
### Leetcode 3. 无重复字符的最长子串
{% highlight C++ %}
/*
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
*/
int lengthOfLongestSubstring(string s) {
    int n = int(s.length());
    if (n == 0) { return 0; }
    int i = 0, j = 0, maxLen = 0;
    unordered_set<char> hash;
    while (j < n) {
        if (hash.count(s[j])) { hash.erase(s[i++]); }
        else {
            hash.insert(s[j++]);
            maxLen = max(maxLen, j - i);
        }
    }
    return maxLen;
}
{% endhighlight %}

### Leetcode 15. 三数之和
{% highlight C++ %}
/*
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素
a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的
三元组。

注意：答案中不可以包含重复的三元组。
*/
vector<vector<int>> threeSum(vector<int>& nums) {
    int n = int(nums.size());
    vector<vector<int>> ans;
    if (n < 3) { return ans; }
    sort(nums.begin(), nums.end());
    for (int i = 0; i < n - 2; i++) {
        int j = i + 1, k = n - 1;
        while (j < k) {
            int sum = nums[i] + nums[j] + nums[k];
            if (sum < 0) { j++; }
            else if (sum > 0) { k--; }
            else {
                while (i < k && nums[i] == nums[i+1]) { i++; }
                while (j < k && nums[j] == nums[j+1]) { j++; }
                while (j < k && nums[k] == nums[k-1]) { k--; }
                ans.push_back({nums[i], nums[j], nums[k]});
                j++;
                k--;
            }
        }
    }
    return ans;
}
{% endhighlight %}

### Leetcode 16. 最接近的三数之和
{% highlight C++ %}
/*
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三
个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在
唯一答案。
*/
int threeSumClosest(vector<int>& nums, int target) {
    int n = int(nums.size()), res = 0, diff = (1ll << 31) - 1;
    if (n < 3) { return 0; }
    sort(nums.begin(), nums.end());
    for (int i = 0; i <= n-3; i++) {
        int j = i + 1, k = n - 1;
        while (j < k) {
            int sum = nums[i] + nums[j] + nums[k];
            if (abs(sum - target) < diff)
            {
                diff = abs(sum - target);
                res = sum;
            }
            if (sum < target) { j++; }
            else if (sum > target) { k--; }
            else { return target; }
        }
    }
    return res;
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

### 位运算符
#### XOR
##### Leetcode 1734. 解码异或后的排列
{% highlight C++ %}
/*
给你一个整数数组 perm ，它是前 n 个正整数的排列，且 n 是个 奇数 。
它被加密成另一个长度为 n - 1 的整数数组 encoded ，满足
encoded[i] = perm[i] XOR perm[i + 1] 。比方说，如果 perm = [1,3,2] ，
那么 encoded = [2,1] 。
给你 encoded 数组，请你返回原始数组 perm 。题目保证答案存在且唯一。

示例 1：
输入：encoded = [3,1]
输出：[1,2,3]
解释：如果 perm = [1,2,3] ，那么 encoded = [1 XOR 2,2 XOR 3] = [3,1]

示例 2：
输入：encoded = [6,5,4,6]
输出：[2,4,1,5,3]
*/
/*
perm包含了1-n+1的所有数字，n为encoded的长度
total = perm[0] ^ ... ^ perm[n-1]
perm[0] ^ perm[1] = encoded[0]
perm[1] ^ perm[2] = encoded[1]
perm[2] ^ perm[3] = encoded[2]
perm[3] ^ perm[4] = encoded[3]
所以，encoded数组中奇数位置的元素XOR，对应了perm1-n的XOR，
由此，可以求得perm[0]的大小
*/
vector<int> decode(vector<int>& encoded) {
    int n = (int)encoded.size();
    vector<int> perm;
    int total = 0;
    for (int i = 1; i <= n+1; i++) { total ^= i; }
    int odd = 0;
    for (int i = 1; i < n; i += 2) { odd ^= encoded[i]; }
    perm.emplace_back(total ^ odd);
    for (int i = 0; i < n; i++) {
        perm.emplace_back(perm.back() ^ encoded[i]);
    }
    return perm;
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
#### 打印杨辉三角

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
<p align="justify">
KMP讲求从左到右依次匹配，当出现误匹配的字符时，将模式串P向右移动若干距离。KMP借助P的前缀（历史匹配信息）构建next数组。将P前缀中与失配字符前最大重合的位置对齐。P中的第一个字符发生失配时，对应的下一个位置应该是-1，表示将整个P移动到失配字符的下一个位置。
$$
\begin{matrix}
a & c & e & a & c & {\color{Red} e} & a & f & d & b & e \\
a & c & e & a & c & {\color{Blue} f} \\
& & & & & \downarrow \\
& & & a & c & e & a & c & f
\end{matrix}
$$
为了更加高效地移动P，next数组表示地下一个位置应当尽可能地小
$$
\begin{matrix}
a & c & f & a & c & {\color{Red} e} & a & f & d & b & e \\
a & c & f & a & c & {\color{Blue} f} \\
& & & & \downarrow & & & \\
& & & a & c & {\color{Blue} f} & a & c & f \\
& & & & \downarrow & & & \\
& & & & & & a & c & f & a & c & f
\end{matrix}
$$
空间复杂度：$O(m)$, 最坏时间复杂度：$O(n + m)$，m是P的长度，n是T的长度
</p>

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
int strStr(string haystack, string needle) {
    int n = (int)haystack.length(), m = (int)needle.length();
    if (m == 0) { return 0; }
    if (n == 0) { return -1; }
    vector<int> next(m, -1);
    int j = 0, k = -1;
    while (j < m - 1) {
        if (k < 0 || needle[j] == needle[k]) {
            j++;
            k++;
            next[j] = needle[j] != needle[k] ? k : next[k];
        }
        else { k = next[k]; }
    }
    int i = 0;
    j = 0;
    while (i < n && j < m) {
        if (j < 0 || haystack[i] == needle[j]) {
            i++;
            j++;
        }
        else { j = next[j]; }
    }
    return i - j > n - m ? -1 : i - j;
}
{% endhighlight %}

##### Leetcode 214. 最短回文串
{% highlight C++ %}
/*
给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。
找到并返回可以用这种方式转换的最短回文串。

示例 1：
输入：s = "aacecaaa"
输出："aaacecaaa"

示例 2：
输入：s = "abcd"
输出："dcbabcd"
*/
string shortestPalindrome(string s) {
    int n = (int)s.length();
    if (n <= 1) { return s; }
    vector<int> next(n, -1);
    int i = 0, pos = -1;
    while (i < n - 1) {
        if (pos < 0 || s[i] == s[pos]) {
            i++;
            pos++;
            next[i] = s[i] != s[pos] ? pos : next[pos];
        }
        else { pos = next[pos]; }
    }
    pos = 0;
    i = n - 1;
    while (i >= 0 && pos < n) {
        if (pos < 0 || s[i] == s[pos]) {
            i--;
            pos++;
        }
        else { pos = next[pos]; }
    }
    string ans = s.substr(pos, n - pos);
    reverse(ans.begin(), ans.end());
    return ans + s;
}
{% endhighlight %}

#### Boyer-Moore
<p align="justify">
1、坏字符<br>
当模式串P与文本串T在某以位置pos发生不匹配时，T[pos]叫做坏字符。<br>
1.1 坏字符哈希表的构造，一遍扫描将P中的字符位置保存在字典中，key为字符本身，value为字符在P中的位置。<br>
1.2 当坏字符不在模式串中，将模式串全部移动到坏字符后一位
$$
\begin{matrix}
a & c & {\color{Red} e} & a & f & d & b & e \\
a & f & {\color{Blue} d} \\
& & & & \downarrow \\
& & & a & f & d
\end{matrix}
$$
1.3 当坏字符出现在模式串中，将P中最近的字符与T中的坏字符对应
$$
\begin{matrix}
a & c & {\color{Red} e} & a & f & d & b & e \\
a & c & {\color{Blue} d} \\
& & & & \downarrow \\
& a & e & d
\end{matrix}
$$
但是如果坏字符在P中的位置大于不匹配的位置j时，移动的距离是负数
$$
\begin{matrix}
& {\color{Red} a} & c & d & a & f & d & b & e \\
& {\color{Blue} b} & a & d \\
& & & & \downarrow \\
{\color{Blue} b} & a & d
\end{matrix}
$$
为了避免这种情况，BM还需要好后缀规则。<br><br>

2、好后缀<br>
因为T与P的匹配是从左到右（移动方向一直是向右），当出现坏字符时，P的后半部分（如果有）一定是匹配好的，这部分叫做好后缀。<br>
2.1 好后缀在P的前缀中出现，将P的好后缀匹配的前缀部分与T对齐<br>
$$
\begin{matrix}
c & a & c & {\color{Red} d} & {\color{Green} e} & {\color{Green} f} & a & d & e & f \\
a & e & f & {\color{Blue} c} & {\color{Green} e} & {\color{Green} f} \\
& & & & \downarrow \\
& & & a & {\color{Green} e} & {\color{Green} f} & c & e & f
\end{matrix}
$$
2.2 好后缀的部分（后缀）在P的前缀中出现，将部分后缀（P前缀中的）与T对齐
$$
\begin{matrix}
c & a & {\color{Red} d} & {\color{Green} c} & {\color{Green} e} & {\color{Green} f} & a & d & e & f \\
a & e & {\color{Blue} f} & {\color{Green} c} & {\color{Green} e} & {\color{Green} f} \\
& & & & \downarrow \\
& & & a & {\color{Green} e} & {\color{Green} f} & c & e & f
\end{matrix}
$$
2.3 好后缀在P的前缀中没有出现，将P整体移动lenP（P的长度）
$$
\begin{matrix}
a & c & {\color{Red} f} & {\color{Green} a} & {\color{Green} f} & d & b & e & c & f\\
& c & {\color{Blue} c} & {\color{Green} a} & {\color{Green} f} \\
& & & & \downarrow \\
& & & & & c & c & a & f
\end{matrix}
$$
2.4 好后缀的构造
$$
\begin{aligned}
& \text{suffix[ k ] = pos} \\
& \text{prefix[ k ] = T/F}
\end{aligned}
$$
k表示好后缀的长度（从1开始），pos表示好后缀在前缀出现的位置（若不存在为-1）, T/F表示长度为k的后缀在P的前缀中是否出现，例如
\begin{aligned}
& \text{P = cefcef} \\
& \text{suffix[1] = 2,} \quad \text{prefix[1] = False} \\
& \text{suffix[2] = 1,} \quad \text{prefix[1] = False} \\
& \text{suffix[3] = 0,} \quad \text{prefix[1] = True} \\
& \text{suffix[4] = -1,} \quad \text{prefix[1] = False} \\
& \text{suffix[5] = -1,} \quad \text{prefix[1] = False} \\
\end{aligned}
3、坏字符与好后缀组合，分别计算两种规则下移动的距离，取较大者。空间复杂度：$O(m + \Sigma)$；时间复杂度：最好情况下$O(\frac{n}{m})$，最差情况下$O(n+m)$
</p>
{% highlight C++ %}
class BoyerMoore{
    vector<int> bc;
    vector<int> suffix;
    bool *prefix;
    string T, P;
    int nT, nP;
public:
    const int N = 256;
    BoyerMoore (string text, string pattern) {
        T = text;
        P = pattern;
        nT = (int)T.length();
        nP = (int)P.length();
    }
    
    ~BoyerMoore () {
        delete []prefix;
    }
    
    void buildBadCharacter() {
        bc.resize(N, -1);
        for (int i = 0; i < nP; i++) {
            bc[P[i]] = i;
        }
    }
    
    void buildGoodSuffix() {
        suffix.resize(nP, -1);
        prefix = new bool [nP]{};
        for (int i = 0; i < nP - 1; i++) {
            // k代表后缀的长度
            int j = i, k = 0;
            while (j >= 0 && P[j] == P[nP - 1 - k]) {
                k++;
                j--;
                suffix[k] = j + 1;
            }
            if (j == -1) {
                prefix[k] = true;
            }
        }
    }
    
    int stepsToMoveWithGoodSuffix(int idxBC) {
        int k = nP - idxBC - 1;
        if (k == 0) { return 0; }
        if (suffix[k] != -1) {
            return idxBC - suffix[k] + 1;
        }
        for (int i = k - 1; i > 0; i--) {
            if (prefix[i]) {
                return nP - i - suffix[i];
            }
        }
        return nP;
    }
    
    vector<int> match() {
        buildBadCharacter();
        buildGoodSuffix();
        vector<int> ans;
        int k = nP - 1;
        while (k < nT) {
            int i = k, j = nP - 1;
            while (j >= 0 && T[i] == P[j]) {
                i--;
                j--;
            }
            if (j == -1) {
                ans.emplace_back(i+1);
                k++;
                continue;
            }
            int stepsWithBC = nP - 1 - bc[T[i]] - (k - i);
            int stepsWithGS = stepsToMoveWithGoodSuffix(j);
            k += max(stepsWithBC, stepsWithGS);
        }
        return ans;
    }
    
    bool isMatch() {
        return !match().empty();
    }
};
/*
string T = "ababab", P = "abab";
// aaabaaabbbabaa,babb: -1
// ababbbbaaabbbaaa,bbbb: 3
BoyerMoore bm(T, P);
vector<int> pos = bm.match();  // [0 2]
*/
{% endhighlight %}

#### Rabin-Karp
<p align="justify">
利用哈希的方法，将字符串转化成一个数。由此，两字符串之间的比较转化为两数字的比较。如果出现散列冲突，逐一比较两字符串是否相等。
</p>
{% highlight C++ %}
int RabinKarp(string T, string P) {
    int n = (int)T.length(), m = (int)P.length();
    if (m == 0) { return 0; }
    if (n == 0) { return -1; }
    int key = 0, value = 0;
    for (int i = 0; i < m - 1; i++) {
        key += P[i];
        value += T[i];
    }
    key += P[m-1];
    for (int i = m - 1, j = -1; i < n; i++) {
        value += T[i];
        if (j >= 0) { value -= T[j]; }
        j++;
        if (value == key) {
            int x = j, y = 0;
            while (x <= i && y < m && T[x] == P[y]) {
                x++;
                y++;
            }
            if (y == m) { return j; }
        }
    }
    return -1;
}

int RabinKarp(string T, string P) {
    int n = (int)T.length(), m = (int)P.length();
    if (m == 0) { return 0; }
    if (n == 0) { return -1; }
    for (int i = 0; i <= n-m; i++) {
        if (T.substr(i, m) == P) {
            return i;
        }
    }
    return -1;
}
{% endhighlight %}