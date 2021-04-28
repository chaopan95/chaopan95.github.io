---
layout: page
title:  "Nowcoder"
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


## 牛客题霸-算法篇
### 19. 子数组的最大累加和问题
{% highlight C++ %}
/*
题目描述
给定一个数组arr，返回子数组的最大累加和
例如，arr = [1, -2, 3, 5, -2, 6, -1]，所有子数组中，[3, 5, -2, 6]
可以累加出最大的和12，所以返回12. 题目保证没有全为负数的数据
时间复杂度为O(n)O(n)，空间复杂度为O(1)O(1)

示例1
输入
[1, -2, 3, 5, -2, 6, -1]
返回值
12
*/
class Solution {
public:
    /**
     * max sum of the subarray
     * @param arr int整型vector the array
     * @return int整型
     */
    int maxsumofSubarray(vector<int>& arr) {
        // write code here
        int curSum = 0, maxSum = 0;
        for (int ele : arr) {
            if (curSum > 0) {
                curSum += ele;
            }
            else { curSum = ele; }
            if (maxSum < curSum) {
                maxSum = curSum;
            }
        }
        return maxSum;
    }
};
{% endhighlight %}

### 35. 最小编辑代价
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
class Solution {
public:
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
};
{% endhighlight %}

### 41. 最长无重复子串
{% highlight C++ %}
/*
题目描述
给定一个数组arr，返回arr的最长无的重复子串的长度(无重复指的是所有数字都不相同)。
示例1
输入
[2,3,4,5]
返回值
4

示例2
输入
[2,2,3,4,3]
返回值
3
*/
class Solution {
public:
    /**
     * 
     * @param arr int整型vector the array
     * @return int整型
     */
    int maxLength(vector<int>& arr) {
        // write code here
        int n = (int)arr.size();
        if (n < 2) { return n; }
        unordered_set<int> hash;
        int i = 0, j = 0, len = 0;
        while (j < n) {
            if (hash.find(arr[j]) == hash.end()) {
                hash.insert(arr[j++]);
                if (len < j - i) { len = j - i; }
            }
            else {
                hash.erase(arr[i++]);
            }
        }
        return len;
    }
};
{% endhighlight %}

### 105. 二分查找-II
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
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
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
};
{% endhighlight %}

### 127. 最长公共子串
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

### 160. 二分查找-I
{% highlight C++ %}
/*
请实现无重复数字的升序数组的二分查找
给定一个 元素有序的（升序）整型数组 nums 和一个目标值 target，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1

示例1
输入
[-1,0,3,4,6,10,13,14],13
返回值
6
说明
13 出现在nums中并且下标为 6
*/
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param nums int整型vector 
     * @param target int整型 
     * @return int整型
     */
    int search(vector<int>& nums, int target) {
        // write code here
        int n = (int)nums.size();
        int l = 0, r = n - 1;
        while (l <= r) {
            int m = (l + r) >> 1;
            if (nums[m] == target) { return m; }
            else if (nums[m] < target) { l = m + 1; }
            else { r = m - 1; }
        }
        return -1;
    }
};
{% endhighlight %}

### 
{% highlight C++ %}

{% endhighlight %}

## 程序员代码面试指南CD
### 3. 不重复打印排序数组中相加和为给定值的所有二元组
{% highlight C++ %}
/*
题目描述
给定排序数组arr和整数k，不重复打印arr中所有相加和为k的不降序二元组
例如, arr = [-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], k = 10，打印结果为：
1, 9
2, 8
时间复杂度为O(n)，空间复杂度为O(1)
输入描述:
第一行有两个整数n, k
接下来一行有n个整数表示数组内的元素
输出描述:
输出若干行，每行两个整数表示答案
按二元组从小到大的顺序输出(二元组大小比较方式为每个依次比较二元组内每个数)
示例1
输入
10 10
-8 -4 -3 0 1 2 4 5 8 9
输出
1 9
2 8
*/
#include<iostream>
#include<vector>
using namespace std;

void printSum(vector<int> arr, int target) {
    int n = int(arr.size());
    int l = 0, r = n - 1;
    while (l < r) {
        int sum = arr[l] + arr[r];
        if (sum < target) { l++; }
        else if (sum > target) { r--; }
        else {
            while (l < r && arr[l] == arr[l+1]) { l++; }
            while (l < r && arr[r] == arr[r-1]) { r--; }
            printf("%d %d\n", arr[l++], arr[r--]);
        }
    }
}


int main(int argc, const char * argv[]) {
    int n = 10, K = 10;
    scanf("%d %d", &n, &K);
    vector<int> arr(n, 0);
    for (int i = 0; i < n; i++) { scanf("%d", &arr[i]); }
    printSum(arr, K);
    return 0;
}
{% endhighlight %}

### 4. 不重复打印排序数组中相加和为给定值的所有三元组
{% highlight C++ %}
/*
题目描述
给定排序数组arr和整数k，不重复打印arr中所有相加和为k的严格升序的三元组
例如, arr = [-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], k = 10，打印结果为：
-4 5 9
-3 4 9
-3 5 8
0 1 9
0 2 8
1 4 5
时间复杂度为O(n^2)，空间复杂度为O(1)

输入描述:
第一行有两个整数n, k
接下来一行有n个整数表示数组内的元素
输出描述:
输出若干行，每行三个整数表示答案
按三元组从小到大的顺序输出(三元组大小比较方式为每个依次比较三元组内每个数)
示例1
输入
10 10
-8 -4 -3 0 1 2 4 5 8 9
输出
-4 5 9
-3 4 9
-3 5 8
0 1 9
0 2 8
1 4 5
*/
#include<iostream>
#include<vector>
using namespace std;

void print3Sum(vector<int> arr, int target) {
    int n = int(arr.size());
    for (int i = 0; i < n - 2; i++) {
        if (arr[i] == arr[i+1]) { continue; }
        int j = i + 1, k = n - 1;
        while (j < k) {
            int sum = arr[i] + arr[j] + arr[k];
            if (sum < target) { j++; }
            else if (sum > target) { k--; }
            else {
                while (j < k && arr[j] == arr[j+1]) { j++; }
                while (j < k && arr[k] == arr[k-1]) { k--; }
                printf("%d %d %d\n", arr[i], arr[j++], arr[k--]);
            }
        }
    }
}


int main(int argc, const char * argv[]) {
    int n = 10, K = 10;
    scanf("%d %d", &n, &K);
    vector<int> arr(n, 0);
    for (int i = 0; i < n; i++) { scanf("%d", &arr[i]); }
    print3Sum(arr, K);
    return 0;
}
{% endhighlight %}

### 17. 机器人达到指定位置方法数
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

### 31. 最长公共子序列
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

### 186. 矩阵的最小路径和
{% highlight C++ %}
/*
题目描述
给定一个 n * m 的矩阵 a，从左上角开始每次只能向右或者向下走，
最后到达右下角的位置，路径上所有的数字累加起来就是路径和，
输出所有的路径中最小的路径和。
输入描述:
第一行输入两个整数 n 和 m，表示矩阵的大小。

接下来 n 行每行 m 个整数表示矩阵。
输出描述:
输出一个整数表示答案。
示例1
输入
4 4
1 3 5 9
8 1 3 4
5 0 6 1
8 8 4 0
输出
12
*/
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, const char *argv[]) {
    int n = 4, m = 4;
    scanf("%d %d", &n, &m);
    vector<vector<int>> dp(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int val = 0;
            scanf("%d", &val);
            if (i == 0 && j == 0) { dp[i][j] = val; }
            else if (i == 0) {
                dp[i][j] = dp[i][j - 1] + val;
            }
            else if (j == 0) {
                dp[i][j] = dp[i - 1][j] + val;
            }
            else {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + val;
            }
        }
    }
    printf("%d\n", dp[n - 1][m - 1]);
    return 0;
}
{% endhighlight %}

### 
{% highlight C++ %}

{% endhighlight %}