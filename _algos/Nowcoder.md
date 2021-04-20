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