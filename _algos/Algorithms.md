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
### 连续子数组最大和、最大积
#### Leetcode 53. 最大子序和
{% highlight C++ %}
/*
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），
返回其最大和。

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
*/
int maxSubArray(vector<int>& nums) {
    int n = int(nums.size());
    int maxSum = -(1ll << 31), curSum = 0;
    for (int i = 0; i < n; i++) {
        if (curSum >= 0) { curSum += nums[i]; }
        else { curSum = nums[i]; }
        if (curSum > maxSum) { maxSum = curSum; }
    }
    return maxSum;
}
{% endhighlight %}

#### Leetcode 152. 乘积最大子数组
{% highlight C++ %}
/*
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
*/
int maxProduct(vector<int>& nums) {
    int n = int(nums.size());
    int minF = nums[0], maxF = nums[0], ans = nums[0];
    for (int i = 1; i < n; i++) {
        int _min = minF, _max = maxF;
        minF = min(nums[i], min(_min * nums[i], _max * nums[i]));
        maxF = max(nums[i], max(_min * nums[i], _max * nums[i]));
        ans = max(ans, maxF);
    }
    return ans;
}
{% endhighlight %}

#### Leetcode 1186. 删除一次得到子数组最大和
{% highlight C++ %}
/*
给你一个整数数组，返回它的某个 非空 子数组（连续元素）在执行一次可选的删除操作后，所能得到
的最大元素总和。

换句话说，你可以从原数组中选出一个子数组，并可以决定要不要从中删除一个元素（只能删一次哦），
（删除后）子数组中至少应当有一个元素，然后该子数组（剩下）的元素总和是所有子数组之中最大的。

注意，删除一个元素后，子数组 不能为空。

请看示例：
示例 1：
输入：arr = [1,-2,0,3]
输出：4
解释：我们可以选出 [1, -2, 0, 3]，然后删掉 -2，这样得到 [1, 0, 3]，和最大。

示例 2：
输入：arr = [1,-2,-2,3]
输出：3
解释：我们直接选出 [3]，这就是最大和。

示例 3：
输入：arr = [-1,-1,-1,-1]
输出：-1
解释：最后得到的子数组不能为空，所以我们不能选择 [-1] 并从中删去 -1 来得到 0。
     我们应该直接选择 [-1]，或者选择 [-1, -1] 再从中删去一个 -1。
*/
class Solution {
public:
    int maximumSum(vector<int>& arr) {
        int n = (int)arr.size();
        if (n == 0) { return 0; }
        const int minEle = *min_element(arr.begin(), arr.end());
        int dp0 = arr[0], dp1 = minEle, ans = dp0;
        for (int i = 1; i < n; i++) {
            int tmp0 = max(dp0 + arr[i], arr[i]);
            int tmp1 = max(dp0, dp1 + arr[i]);
            dp0 = tmp0;
            dp1 = tmp1;
            ans = max(ans, max(dp0, dp1));
        }
        return ans;
    }
};
{% endhighlight %}

### 背包问题
#### 01背包
<p align="justify">
一个背包有一定的承重cap，有N件物品，每件都有自己的价值，记录在数组v中，也都有自己的重量，记录在数组w中，每件物品只能选择要装入背包还是不装入背包，要求在不超过背包承重的前提下，选出物品的总价值最大。给定物品的重量w价值v及物品数n和承重cap。请返回最大总价值。<br>
测试样例：<br>
[1,2,3],[1,2,3],3,6<br>
返回：6<br><br>

动态规划：dp[i][j]表示前i件物品在最大重量j的条件下的价值
$$
dp[i][j] =\max
\begin{cases}
dp[i-1]][j], &\quad \text{we don't put i-th item in our bag} \\
dp[i-1][j - w[i-1]], &\quad \text{otherwise, but } j > w[i-1], i = 1, 2, ..., N
\end{cases}
$$
</p>
{% highlight C++ %}
/* *
 * v[]: value for each item
 * w[]: weight for each item
 * W: capacity limit
 * N: number of item
*/
int knapsack(int v[], int w[], int W, int N) {
    int **dp = new int *[N+1];
    for (int i = 0; i <= N; i++) { dp[i] = new int [W+1]{}; }
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= W; j++) {
            if (j < w[i-1]) { dp[i][j] = dp[i-1][j]; }
            else {
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i-1]] + v[i-1]);
            }
        }
    }
    int res = dp[N][W];
    for (int i = 0; i <= N; i++) { delete []dp[i]; }
    delete []dp;
    return res;
}
{% endhighlight %}

#### Leetcode 416. 分割等和子集
{% highlight C++ %}
/*
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割
成两个子集，使得两个子集的元素和相等。

示例 1：
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2：
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
*/
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = (int)nums.size();
        if (n <= 1) { return false; }
        int sum = 0;
        for (int &num : nums) { sum += num; }
        if (sum % 2 == 1) { return false; }
        int target = sum / 2;
        bool *dp = new bool [target+1]{};
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= nums[i]; j--) {
                if (i == 0) { dp[j] = (j == nums[i]); }
                else if (j == 0) { dp[j] = true; }
                else {
                    dp[j] = dp[j] || dp[j-nums[i]];
                }
            }
        }
        bool ans = dp[target];
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

### 编辑距离
#### Leetcode 72. 编辑距离
{% highlight C++ %}
/*
给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符

输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
*/
int minDistance(string word1, string word2) {
    string s1 = word1, s2 = word2;
    int n1 = int(s1.length()), n2 = int(s2.length());
    int **dp = new int *[n1+1];
    for (int i = 0; i <= n1; i++) { dp[i] = new int [n2+1]{}; }
    for (int i = 0; i <= n1; i++) {
        for (int j = 0; j <= n2; j++) {
            if (i == 0 || j == 0) { dp[i][j] = max(i, j); }
            else
            {
                int a = dp[i-1][j] + 1;
                int b = dp[i][j-1] + 1;
                int c = dp[i-1][j-1] + (s1[i-1] != s2[j-1]);
                dp[i][j] = min(min(a, b), c);
            }
        }
    }
    int ans = dp[n1][n2];
    for (int i = 0; i <= n1; i++) { delete []dp[i]; }
    delete []dp;
    return ans;
}
{% endhighlight %}

#### 牛客题霸-算法篇 35. 最小编辑代价
{% highlight C++ %}
/*
题目描述
给定两个字符串str1和str2，再给定三个整数ic，dc和rc，分别代表插入、
删除和替换一个字符的代价，请输出将str1编辑成str2的最小代价。
示例1
输入
"abc","adc",5,3,2
返回值
2

示例2
输入
"abc","adc",5,3,100
返回值
8
*/
/**
 * min edit cost
 * @param str1 string字符串 the string
 * @param str2 string字符串 the string
 * @param ic int整型 insert cost
 * @param dc int整型 delete cost
 * @param rc int整型 replace cost
 * @return int整型
 */
int minEditCost(string str1, string str2, int ic, int dc, int rc) {
    // write code here
    int n1 = (int)str1.length(), n2 = (int)str2.length();
    vector<vector<int>> dp(n1+1, vector<int>(n2+1, 0));
    for (int i = 0; i <= n1; i++) {
        for (int j = 0; j <= n2; j++) {
            if (i == 0) { dp[i][j] = ic * j; }
            else if (j == 0) { dp[i][j] = dc * i; }
            else {
                int ist = dp[i][j-1] + ic;
                int dlt = dp[i-1][j] + dc;
                int rpl = dp[i-1][j-1] + rc * (str1[i-1] != str2[j-1]);
                dp[i][j] = min(ist, min(dlt, rpl));
            }
        }
    }
    return dp[n1][n2];
}
{% endhighlight %}

### 最长公共子序列
#### Leetcode 1143. 最长公共子序列
{% highlight C++ %}
/*
给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列
的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变
字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
*/
int longestCommonSubsequence(string text1, string text2) {
    string s1 = text1, s2 = text2;
    int n1 = int(s1.length()), n2 = int(s2.length());
    int **dp = new int *[n1+1];
    for (int i = 0; i < n1+1; i++) { dp[i] = new int [n2+1]{}; }
    for (int i = 1; i < n1+1; i++) {
        for (int j = 1; j < n2+1; j++) {
            if (s1[i-1] == s2[j-1]) { dp[i][j] = dp[i-1][j-1] + 1; }
            else {
                dp[i][j] = (dp[i-1][j] > dp[i][j-1] ?
                            dp[i-1][j] : dp[i][j-1]);
            }
        }
    }
    int lcs = dp[n1][n2];
    for (int i = 0; i < n1+1; i++) { delete []dp[i]; }
    delete []dp;
    return lcs;
}
{% endhighlight %}

#### 程序员代码面试指南 31. 最长公共子序列
{% highlight C++ %}
/*
题目描述
给定两个字符串str1和str2，输出连个字符串的最长公共子序列。如过最长公共子序列为空，则输出-1。
输入描述:
输出包括两行，第一行代表字符串str1，第二行代表str2。
输出描述:
输出一行，代表他们最长公共子序列。如果公共子序列的长度为空，则输出-1。
示例1
输入
1A2C3D4B56
B1D23CA45B6A
输出
123456
说明
"123456"和“12C4B6”都是最长公共子序列，任意输出一个。
*/
#include<iostream>
#include<string>
#include<vector>
using namespace std;

string LCS(string s1, string s2) {
    int n1 = (int)s1.length(), n2 = (int)s2.length();
    vector<vector<int>> dp(n1+1, vector<int>(n2+1, 0));
    string ans = "";
    for (int i = n1 - 1; i >= 0; i--) {
        for (int j = n2 - 1; j >= 0; j--) {
            if (s1[i] == s2[j]) { dp[i][j] = dp[i+1][j+1] + 1; }
            else { dp[i][j] = max(dp[i+1][j], dp[i][j+1]); }
        }
    }
    int i = 0, j = 0;
    while (i < n1 && j < n2) {
        if (s1[i] == s2[j]) {
            ans.push_back(s1[i]);
            i++;
            j++;
        }
        else if (dp[i+1][j] > dp[i][j+1]) {
            i++;
        }
        else { j++; }
    }
    return dp[0][0] ? ans : "-1";
}

int main(int argc, const char * argv[]) {
    // insert code here...
    string s1 = "1A2C3D4B56", s2 = "B1D23CA45B6A";
    cin >> s1 >> s2;
    printf("%s\n", LCS(s1, s2).c_str());
    return 0;
}
{% endhighlight %}

### 最长公共子串
#### Leetcode 718. 最长重复子数组
{% highlight C++ %}
/*
给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

Input:
A: [1,2,3,2,1]
B: [3,2,1,4,7]
Output: 3
Explanation: 
The repeated subarray with maximum length is [3, 2, 1].
*/
int findLength(vector<int>& A, vector<int>& B) {
    int na = int(A.size()), nb = int(B.size());
    vector<vector<int>> dp(na+1, vector<int>(nb+1, 0));
    int ans = 0;
    for (int i = 1; i <= na; i++) {
        for (int j = 1; j <= nb; j++) {
            if (A[i-1] == B[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                ans = max(ans, dp[i][j]);
            }
        }
    }
    return ans;
}
{% endhighlight %}

#### 牛客题霸-算法篇 127. 最长公共子串
{% highlight C++ %}
/*
给定两个字符串str1和str2,输出两个字符串的最长公共子串
题目保证str1和str2的最长公共子串存在且唯一。

示例1
输入
"1AB2345CD","12345EF"
返回值
"2345"
*/
class Solution {
public:
    /**
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    string LCS(string str1, string str2) {
        // write code here
        int n1 = (int)str1.length(), n2 = (int)str2.length();
        vector<vector<int>> dp(n1+1, vector<int> (n2+1, 0));
        int len = 1, pos = 0;
        for (int i = n1 - 1; i >= 0; i--) {
            for (int j = n2 - 1; j >= 0; j--) {
                if (str1[i] == str2[j]) {
                    dp[i][j] = dp[i+1][j+1] + 1;
                    if (len < dp[i][j]) {
                        len = dp[i][j];
                        pos = i;
                        
                    }
                }
            }
        }
        return str1.substr(pos, len);
    }
};
{% endhighlight %}

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

### 棋盘形DP
#### Leetcode 62. 不同路径
<p align="justify">
$$
\begin{aligned}
& dp[i][1] = 1, \quad i = 1, 2, ..., m \\
& dp[1][j] = 1, \quad j = 1, 2, ..., n \\
& dp[i][j] = dp[i-1][j] + dp[i][j-1], \quad i = 2, 3, ..., m \quad j = 2, 3, ..., n
\end{aligned}
$$
</p>
{% highlight C++ %}
/*
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the
bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
*/
int uniquePaths(int m, int n) {
    int **dp = new int *[m];
    for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
    for (int i = 0; i < m; i++) { dp[i][0] = 1; }
    for (int j = 0; j < n; j++) { dp[0][j] = 1; }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    int ans = dp[m-1][n-1];
    for (int i = 0; i < m; i++) { delete []dp[i]; }
    delete []dp;
    return ans;
}
{% endhighlight %}

#### Leetcode 63. 不同路径II
{% highlight C++ %}
/*
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
*/
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    vector<vector<int>> mat = obstacleGrid;
    int m = int(mat.size());
    if (m == 0) { return 0; }
    int n = int(mat[0].size());
    if (n == 0) { return 0; }
    int **dp = new int *[m];
    for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
    if (mat[0][0]) { return 0; }
    dp[0][0] = 1;
    for (int i = 1; i < m; i++) {
        if (mat[i][0]) { dp[i][0] = 0; }
        else { dp[i][0] = dp[i-1][0]; }
    }
    for (int j = 1; j < n; j++) {
        if (mat[0][j]) { dp[0][j] = 0; }
        else { dp[0][j] = dp[0][j-1]; }
    }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (mat[i][j]) { dp[i][j] = 0; }
            else { dp[i][j] = dp[i-1][j] + dp[i][j-1]; }
        }
    }
    int ans = dp[m-1][n-1];
    for (int i = 0; i < m; i++) { delete []dp[i]; }
    delete []dp;
    return ans;
}
{% endhighlight %}

#### Leetcode 63. 最小路径和
{% highlight C++ %}
/*
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角
到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

1 3 1
1 5 1
4 2 1
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
*/
int minPathSum(vector<vector<int>>& grid) {
    int m = int(grid.size());
    if (m == 0) { return 0; }
    int n = int(grid[0].size());
    if (n == 0) { return 0; }
    int **dp = new int *[m];
    for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; i++) { dp[i][0] = dp[i-1][0] + grid[i][0]; }
    for (int j = 1; j < n; j++) { dp[0][j] = dp[0][j-1] + grid[0][j]; }
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
        }
    }
    int ans = dp[m-1][n-1];
    for (int i = 0; i < m; i++) { delete []dp[i]; }
    delete []dp;
    return ans;
}
{% endhighlight %}

#### 程序员代码面试指南 17. 机器人达到指定位置方法数
{% highlight C++ %}
/*
题目描述
假设有排成一行的N个位置，记为1~N，开始时机器人在M位置，机器人可以往左或者往右走，
如果机器人在1位置，那么下一步机器人只能走到2位置，如果机器人在N位置，那么下一步机
器人只能走到N-1位置。规定机器人只能走k步，最终能来到P位置的方法有多少种。由于方
案数可能比较大，所以答案需要对1e9+7取模。
输入描述:
输出包括一行四个正整数N（2<=N<=5000）、M(1<=M<=N)、K(1<=K<=5000)、P(1<=P<=N)。
输出描述:
输出一个整数，代表最终走到P的方法数对10^9+7取模后的值。
示例1
输入
复制
5 2 3 3
输出
复制
3
说明
1).2->1,1->2,2->3
2).2->3,3->2,2->3
3).2->3,3->4,4->3
*/
#include<iostream>
#include<vector>
using namespace std;

int moveRobot(int N, int M, int K, int P) {
    const int MOD = 1000000007;
    vector<vector<int>> dp(N+2, vector<int>(2, 0));
    dp[M][0] = 1;
    while (K--) {
        for (int i = 1; i <= N; i++) {
            dp[i][1] = (dp[i-1][0] + dp[i+1][0]) % MOD;
        }
        for (int i = 1; i <= N; i++) {
            dp[i][0] = dp[i][1];
        }
    }
    return dp[P][1];
}


int main(int argc, const char *argv[]) {
    int N = 5, M = 2, K = 3, P = 3;
    scanf("%d %d %d %d", &N, &M, &K, &P);
    printf("%d\n", moveRobot(N, M, K, P));
    return 0;
}
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
1、坏字符<br><br>

2、好后缀
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
{% highlight C++ %}

{% endhighlight %}