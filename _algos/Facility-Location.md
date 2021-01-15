---
layout: post
title:  "Facility Location"
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


## 1. Description
<p align="justify">
A finite client set $\mathbb{D}$, a finite factory set $\mathbb{F}$, a fixed cost $f_{i} \in \mathbb{R}_{+}$ for opening a factory, a capacity $K_{i}$ for each factory $i \in \mathbb{F}$, a service cost $c_{ij} \in \mathbb{R}_{+}$ offered by factory $i \in \mathbb{F}$ to client $j \in \mathbb{D}$<br><br>

We hope to find a subset $X \subseteq \mathbb{F}$ (opened factory) with a minimum cost
$$\sum_{i \in X} f_{i} + \sum_{j \in \mathbb{D}} c_{\sigma(j) j}$$

$\sigma(j)$ debote a map from a client to a factory. Note each factory has a limited capacity.
</p>

## 2. Solution
<p align="justify">
$$
\begin{aligned}
\min & \quad \sum_{i \in \mathbb{F}}f_{i}y_{i} + \sum_{i \in \mathbb{F}}\sum_{j \in \mathbb{D}} c_{ij} x_{ij} \\
\text{s.c.} & \quad \sum_{j \in \mathbb{D}} x_{ij} \leq K_{i} y_{i} \quad & i \in \mathbb{F} \\
& \quad \sum_{i \in \mathbb{F}} x_{ij} = 1 \quad & j \in \mathbb{D} \\
& \quad x_{ij} \in \{0, 1\} \quad & i \in \mathbb{F}, j \in \mathbb{D} \\
& \quad y_{i} \{0, 1\} \quad & i \in \mathbb{F}
\end{aligned}
$$
</p>
{% highlight C++ %}

{% endhighlight %}