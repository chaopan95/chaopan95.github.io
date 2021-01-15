---
layout: post
title:  "Greatest Common Divisor"
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


## 1. Greatest Common Divisor
### 1.1 Two numbers
#### 1.1.1 Enumerate
<p align="justify">
Brutal force: enumerate all numbers from 1 to min(a, b), save the greatest.
</p>
{% highlight C++ %}
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
{% endhighlight %}
<p align="justify">
<br>
</p>

#### 1.1.2 Euclidean Algorithm
{% highlight C++ %}
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
<p align="justify">
<br>
</p>

### 1.2 A vector of numbers
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
<p align="justify">
<br>
</p>


## 2. Least Common Multiple 
<p align="justify">
Suppose we have two numbers a and b, their least common multiple is
$$\text{LCM} = \frac{a * b}{\text{GCD}(a, b)}$$
</p>
{% highlight C++ %}
int LCM(int a, int b)
{
    int gcd = GCD_Euclidean_Algorithm_divide(a, b);
    return (a * b) / gcd;
}
{% endhighlight %}
<p align="justify">
<br>
</p>
