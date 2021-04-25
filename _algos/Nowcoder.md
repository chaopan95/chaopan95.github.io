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

## 
{% highlight C++ %}

{% endhighlight %}

## 程序员代码面试指南CD
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



## 
{% highlight C++ %}

{% endhighlight %}