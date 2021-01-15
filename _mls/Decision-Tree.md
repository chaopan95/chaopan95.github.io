---
layout: post
title:  "Decision Tree"
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


## 1. Decision Tree
### 1.1 ID3
<p align="justify">
$$
\begin{aligned}
& H(p) = -\sum_{i=1}^{N} p_{i} \log p_{i} \\
& H(D) = -\sum_{k=1}^{K} \frac{C_{k}}{D} \log \frac{C_{k}}{D} \\
& H(D \mid A) = \sum_{i=1}^{N} \frac{D_{i}}{D} H(D_{i}) = -\sum_{i=1}^{N} \frac{D_{i}}{D} \sum_{k=1}^{K} \frac{D_{ik}}{D_{i}} \log \frac{D_{ik}}{D_{i}} \\
& g(D, A) = H(D) - H(D \mid A)
\end{aligned}
$$
</p>

### 1.2 C4.5
<p align="justify">
$$g_{R}(D, A) = \frac{H(D) - H(D \mid A)}{H(D)}$$
</p>

### 1.3 CART
<p align="justify">
$$
\begin{aligned}
& \text{Gini}(p) = \sum_{k=1}^{K} p_{k} (1 - p_{k}) = 1 - \sum_{k=1}^{K} p_{k}^{2} \\
& \text{Gini}(p) = 2p(1-p) \\
& \text{Gini}(D) = 1 - \sum_{k=1}^{K} (\frac{C_{k}}{D})^{2} \\
& D_{1} = \{(x, y) \in D \mid A(x) = a\}, \quad D_{2} = D - D_{1} \\
& \text{Gini}(D, A) = \frac{D_{1}}{D} \text{Gini}(D_{1}) + \frac{D_{2}}{D} \text{Gini}(D_{2})
\end{aligned}
$$
</p>

## 2. Random Forest
## 3. GBDT
