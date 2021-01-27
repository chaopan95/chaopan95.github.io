---
layout: post
title:  "Edit Distance"
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


## 1. Levenshtein distance
<p align="justify">
$$
\text{LD}_{a, b}(i, j) =
\begin{cases}
\max(i, j), & \min(i, j) = 0 \\
\min
\begin{cases}
\text{LD}_{a, b}(i-1, j) + 1 \\
\text{LD}_{a, b}(i, j-1) + 1 \\
\text{LD}_{a, b}(i-1, j-1) + 1_{a_{i} \neq b_{j}}
\end{cases}, & \text{otherwise}
\end{cases}
$$
a, b denote two strings to compare.<br>
If $\text{LD}_{a, b}(i, j) = \text{LD}_{a, b}(i-1, j) + 1$, delete $a_{i}$<br>
If $\text{LD}_{a, b}(i, j) = \text{LD}_{a, b}(i, j-1) + 1$, insert $b_{j}$ at $a_{i}$<br>
If $\text{LD}_{a, b}(i, j) = \text{LD}_{a, b}(i-1, j-1) + 1_{a_{i} \neq b_{j}}$, replace $a_{i}$ with $b_{j}$
</p>
{% highlight C++ %}
int min(int a, int b, int c)
{
    int d = a < b ? a : b;
    return d < c ? d : c;
}
int editDistance(string word1, string word2) {
    // write code here
    int n1 = int(word1.length()), n2 = int(word2.length());
    int **dp = new int *[n1+1];
    for (int i = 0; i <= n1; i++)
    {
        dp[i] = new int [n2+1]{};
        dp[i][0] = i;
    }
    for (int j = 0; j <= n2; j++) { dp[0][j] = j; }
    for (int i = 1; i <= n1; i++)
    {
        for (int j = 1; j <= n2; j++)
        {
            int a = dp[i-1][j] + 1;
            int b = dp[i][j-1] + 1;
            int c = dp[i-1][j-1] + (word1[i-1] != word2[j-1]);
            dp[i][j] = min(a, b, c);
        }
    }
    int res = dp[n1][n2];
    for (int i = 0; i <= n1; i++) { delete []dp[i]; }
    delete []dp;
    return res;
}
{% endhighlight %}

## 2. Damerau-Levenshtein distance
<p align="justify">
$$
\text{DLD}_{a, b}(i, j) =
\begin{cases}
\max(i, j), & \quad \text{if } \min(i, j) = 0 \\
\min
\begin{cases}
\text{DLD}_{a, b}(i-1, j) + 1 \\
\text{DLD}_{a, b}(i, j-1) + 1 \\
\text{DLD}_{a, b}(i-1, j-1) + 1_{a_{i} \neq b_{j}} \\
\text{DLD}_{a, b}(i-2, j-2) + 1
\end{cases}, & \quad \text{if } i, j > 1 \text{ and } a_{i} = b_{j-1} \text{ and } a_{i-1} = b_{j} \\
\min
\begin{cases}
\text{DLD}_{a, b}(i-1, j) + 1 \\
\text{DLD}_{a, b}(i, j-1) + 1 \\
\text{DLD}_{a, b}(i-1, j-1) + 1_{a_{i} \neq b_{j}}
\end{cases}, & \quad \text{otherwise}
\end{cases}
$$
</p>
{% highlight C++ %}

{% endhighlight %}