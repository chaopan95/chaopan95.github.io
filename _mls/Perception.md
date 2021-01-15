---
layout: post
title:  "Perception"
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


## 1. Definition
<p align="justify">
Perception is a classification model. Input space $X = \{x_{1}, x_{2}, ..., x_{n}\}$, output space $Y = \{+1 , -1\}$. Discriminant function
$$
f(x_{i}) = \text{sign}(w \cdot x_{i} + b), \quad \text{where } \text{sign}(x_{i}) =
\begin{cases}
1, \quad x_{i} \geq 0 \\
-1, \quad x_{i} < 0
\end{cases}
$$
$W \cdot X$ represents an inner product.<br><br>
</p>

## 2. Algorithm
<p align="justify">
Dataset T = $\{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{n}, y_{n})\}$<br><br>

If one point ($x_{i}$, $y_{i}$) is misclassified
$$y_{i} \times f(x_{i}) = y_{i} \times \text{sign}(x_{i}) = -1$$

Cost funtion
$$L(w, b) = -\sum_{(x_{i}, y_{i}) \in M} y_{i}(w \cdot x_{i} + b), \quad \text{where } M \text{ is a set of misclassified points}$$

Gradient descent
$$
\begin{aligned}
& \frac{\partial L(w, b)}{\partial w} = -\sum_{(x_{i}, y_{i}) \in M} y_{i} x_{i} \\
& \frac{\partial L(w, b)}{\partial b} = -\sum_{(x_{i}, y_{i}) \in M} y_{i}
\end{aligned}
$$

Parameters update by picking up one misclassified point
$$
\begin{aligned}
& w := w - \eta \frac{\partial L(w, b)}{\partial w} \\
& b := b - \eta \frac{\partial L(w, b)}{\partial b}
\end{aligned} \Rightarrow
\begin{aligned}
& w := w - \eta (-y_{i} x_{i}) \\
& b := b - \eta (-y_{i})
\end{aligned} \Rightarrow
\begin{aligned}
& w := w + \eta y_{i} x_{i} \\
& b := b + \eta y_{i}
\end{aligned}, \quad \text{where } \eta \in (0, 1]
$$

<b>Algorithm</b><br>
<b>01</b>. Input: dataset T = $\{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{n}, y_{n})\}$<br>
<b>02</b>. Output: $w^{*}$, $b^{*}$<br>
<b>03</b>. Initialise w, b with 0<br>
<b>04</b>. While True do:<br>
<b>05</b>. &emsp;Calculate misclassified set M<br>
<b>06</b>. &emsp;If $M = \varnothing$:<br>
<b>07</b>. &emsp;&emsp;break<br>
<b>08</b>. &emsp;Pick up one point form M, update w, b<br>
<b>09</b>. return w, b
</p>
{% highlight Python %}
import numpy as np

def sign(x):
    res = x.copy()
    idx = x >= 0
    res[:] = -1
    res[idx] = 1
    return res

def Perception(T, eta=0.1):
    X = T[:, :-1]
    Y = T[:, -1]
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    while True:
        y_hat = sign(np.dot(X, w) + b)
        M = np.nonzero(Y != y_hat)[0]
        if len(M) == 0: break
        pickedUp = np.random.choice(M, 1)
        w = w + eta * Y[pickedUp] * X[pickedUp, :][0]
        b = b + eta * Y[pickedUp]
    return w, b

T = [[3, 3, 1],
     [4, 3, 1],
     [1, 1, -1]]
T = np.array(T)
w, b = Perception(T)
{% endhighlight %}
<p align="justify">
<b>Novikoff</b><br>
If dataset T is linearly seperable,<br>
(i) there exists a hyperplan $w^{*} \cdot x + b^{*} = 0$, which is able to discriminate all positive instances and negative instances, i.e. for all instances $(x_{i}, y_{i}), i = 1, 2, ..., n$, there exists a $\gamma > 0$
$$y_{i} (w^{*} \cdot x_{i} + b^{*}) \geq \gamma$$
(ii) Moreover, set $R = \max_{i = 1, 2, ..., n} \left \| x_{i} \right \|$, the number of misclassification k is
$$k \leq (\frac{R}{\gamma})^{2}$$
$\bigstar$ Prove<br>
1) all instances $x_{i}$ can be classified correctly
$$y_{i} (w^{*} \cdot x + b^{*}) > 0$$
So, we can find a $\gamma$ to satisfy (i)
$$\gamma = \min_{i = 1, 2, ..., n} y_{i} (w^{*} \cdot x_{i} + b^{*})$$
2) suppose in $k^{\text{th}}$ iteration, we have parameters
$$
W_{k} =
\begin{bmatrix}
w_{k} \\
b_{k} 
\end{bmatrix}
$$
In last iteration, we update $w_{k-1}$ and $b_{k-1}$
$$
W_{k} =
\begin{bmatrix}
w_{k} \\
b_{k} 
\end{bmatrix} =
\begin{bmatrix}
w_{k-1} + \eta y_{i} x_{i} \\
b_{k-1} + \eta y_{i}
\end{bmatrix} =
\begin{bmatrix}
w_{k-1} \\
b_{k-1}
\end{bmatrix} + \eta y_{i}
\begin{bmatrix}
x_{i} \\
1
\end{bmatrix}
$$

Two inequalities
$$
\begin{aligned}
w_{k} \cdot w^{*} & = (w_{k} + \eta y_{i} x_{i}) \cdot w^{*} \\
& \geq w_{k-1} \cdot w^{*} + \eta \gamma \\
& \geq w_{k-2} \cdot w^{*} + 2 \eta \gamma \\
& \geq k \eta \gamma
\end{aligned}
$$
$$
\begin{aligned}
\left \| w_{k} \right \|^{2} & = (w_{k-1} + \eta y_{i} x_{i})^{2} \\
& = \left \| w_{k-1} \right \|^{2} + 2 w_{k-1} \eta y_{i} x_{i} + \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq \left \| w_{k-1} \right \|^{2} + \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq \left \| w_{k-1} \right \|^{2} + \eta^{2} R^{2} \\
& \leq k \eta^{2} \left \| x_{i} \right \|^{2} \\
& \leq k \eta^{2} R^{2}
\end{aligned}
$$
So,
$$
\begin{aligned}
& k \eta \gamma \leq w^{k} \cdot w^{k} \leq \left \| w_{k} \right \| \left \| w^{*} \right \| \leq \sqrt{k} \eta R \\
& k \leq (\frac{R}{\gamma})^{2}
\end{aligned}
$$
</p>


## 3. Evaluation
<p align="justify">
<b>Pros and Cons</b><br>
$\bigstar$ pros<br>
-- easy to implement<br>

$\bigstar$ cons<br>
-- cannot handle non-linear classification, e.g. XOR<br>

</p>
