---
layout: page
title:  "Data Structures and Algorithms"
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


<p align="justify">
<a href="https://www.coursera.org/account/accomplishments/specialization/certificate/SUNWVP6REHEU"> My certificate.</a><br>
</p>


## 1. Introduction
<p align="justify">
Computer science should be called computing science, for the same reason why surgery is not called knife science. -- E. Dijkstra.<br><br>

Algorithms + Data structures = Programs -- N. Wirth<br><br>

To measure is to know. If you can do not measure it, you can not improve it. -- Lord Kelvin
</p>

### 1.1 Algorithms
<p align="justify">
An algorithm is a computing method for a specific question: input, ouput, accurate, determinated, feasible, finite.<br><br>

A good algoritm is accurate, robust, readable and efficient.
</p>

### 1.2 Turing Machine
<p align="justify">
Turing machine has a tape on which there are finite alphabets. Besides, there is a movale head along the tape to read or write alphabets. We can use a transition function to describe states of Turing machine.
$$(q, c; d, L/R, p)$$

Where current state q, current alphabet c, replace c with d, move left or right a cell to arrive at a new state p. If turing machine arrives at h state, stop.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_2_Turing_Machine_1.png"/></center>
</p>

### 1.3 Random Access Machine
<p align="justify">
RAM has unlimited registers in roder.
$$R[0], R[1], R[2], \cdots$$

Each basic operation in $O(1)$.
$$
\begin{matrix}
R[ i ]<-c &  & R[ i ]<-R[R[ j ]] & & R[ i ]<-R[ i ]+R[ k ]\\
R[ i ]<-R[ j ] &  & R[R[ i ]]<-R[ j ] &  & R[ i ]<-R[ i ]+R[ k ]\\
\end{matrix}
$$

IF R[ i ] = 0 GOTO 1 etc.<br><br>

Algorithm's time complexity is equal to a number of basic operations in this model by ignoring effect of different hardware.
</p>

### 1.4 Notation
<p align="justify">
T(n) is computing cost for a problem with n size.
$$T(n) = max\{ T(P) | \left | P \right | =n \}$$
</p>

#### 1.4.1 Big-O Notation
<p align="justify">
T(n) = O(f(n)) iff $\exists$ c > 0, when n >> 2, we have T(n) < cf(n). For example
$$\sqrt{5n[3n(n+2)+4]+6} < \sqrt{5n[6n^{2}+4]+6} < \sqrt{35n^{2}+6} < 6n^{1.5} = O(n^{1.5})$$

2-subset is a NP-complete problem.
</p>

#### 1.4.2 Big-$\Omega$ Notation
<p align="justify">
T(n) = $\Omega$(f(n)):<br>
$\exists$ c, when n >> 2, we have T(n) > cf(n)
</p>

#### 1.4.3 Big-$\Theta$ Notation
<p align="justify">
T(n) = $\Theta$(f(n)):<br>
$\exists$ $c_{1}$ > $c_{2}$ > 0, when n >> 2, we have $c_{1}$f(n) > T(n) > $c_{2}$f(n)
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_4_3_Big_Theta_1.png"/></center>
</p>

### 1.5 Series
<p align="justify">
Arithmetic series
$$T(n) = 1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2} = O(n^{2})$$

Power series
$$\sum_{k=0}^{n}k^{d} \approx \int_{0}^{n}x^{d+1}dx = \frac{1}{d+1}x^{d+1}|_{0}^{n} = \frac{1}{d+1}n^{d+1} = O(n^{d+1})$$

$$T_{2}(n) = 1^{2} + 2^{2} + 3^{2} + \cdots + n^{2} = \frac{n(n+1)(2n+1)}{6} = O(n^{3})$$

$$T_{3}(n) = 1^{3} + 2^{3} + 3^{3} + \cdots + n^{3} = \frac{n^{2}(n+1)^{2}}{4} = O(n^{4})$$

$$T_{4}(n) = 1^{4} + 2^{4} + 3^{4} + \cdots + n^{4} = \frac{n(n+1)(2n+1)(3n^{2}+3n-1)}{30} = O(n^{5})$$

Geometric series
$$T_{a}(n) = a^{0} + a^{1} + a^{2} + \cdots +a^{n} = \frac{a^{n+1}-1}{a-1} = O(a^{n})$$

$$1 + 2 + 4 + \cdots + 2^{n} = O(2^{n+1}) = O(2^{n})$$

Converged series
$$\frac{1}{\frac{1}{2}} + \frac{1}{\frac{2}{3}} + \frac{1}{\frac{3}{4}} + \cdots + \frac{1}{\frac{n-1}{n}} = 1 - \frac{1}{n} = O(1)$$

$$1 + \frac{1}{2^{2}} + \cdots + \frac{1}{n^{2}} < 1 + \frac{1}{2^{2}} + \cdots  = \frac{\pi^{6}}{6} = O(1)$$

$$\frac{1}{3} + \frac{1}{7} + \frac{1}{8} + \frac{1}{35} + \frac{1}{24} + \frac{1}{26} + \frac{1}{31} + \frac{1}{35} + \cdots = 1 = O(1)$$

Harmonic series
$$1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n} = \Theta(logn)$$

Logarithmic series
$$log1 + log2 + log3 + \cdots + logn = log(n!) = \Theta(nlogn)$$
</p>

### 1.6 Recurrence
#### 1.6.1 Recursion Tracking
<p align="justify">
T(n) = $\sum$ O(single recursion instance), is helpful for simple recursion.
</p>

#### 1.6.2 Recursion Equation
<p align="justify">
We have an implicite recursive equation and an initial value, we need to solve its explicite equation. For example
$$T(n) = T(n-1) + O(1) \quad T(0) = O(1)$$
</p>

#### 1.6.3 Decrease and Conquer
<p align="justify">
If a problem is complicated, we can reduce its size by splitting it into a smaller problem (sub-problem) and a normal problem (e.g. size in O(1)).<br><br>

For example, given an array A, get its inverse order.
</p>
{% highlight C++ %}
void reverse(int *A, int lo, int hi)
{
	if (lo < hi)
	{
		swap(A[lo], A[hi]);
		reverse(A[*A, lo+1, hi-1]);
	}
	else
	{
		return; //base case
	}
}
{% endhighlight %}

#### 1.6.4 Divide and Conquer
<p align="justify">
Split a big problem into several (usually 2) sub-problem woth a same size.<br><br>

For example, give an array A, get its sum.
</p>
{% highlight C++ %}
int sum(int A[], int lo, int hi)
{
	if (lo == hi)
	{
		return A[lo]; //base case
	}
	int mi = (lo + hi) >> 1;
	return sum(A, lo, mi) + sum(A, mi+1, hi);
}
{% endhighlight %}
<p align="justify">
solve recursive equation
$$
\begin{aligned}
& T(n) = 2T(\frac{n}{2}) + O(1) \quad T(1) = O(1) \\
& T(n) = 2T(\frac{n}{2}) + c_{1} \\
& T(n) + c_{1} = 2(T(\frac{n}{2}) + c_{1}) = 2^{2}(T(\frac{n}{4}) + c_{1}) = \cdots = 2^{logn}(T(1)+c_{1}) = n(c_{2} + c_{1}) \\
& T(n) = O(n)
\end{aligned}
$$
</p>

### 1.7. Dynamic programming
#### 1.7.1 Fibonacci
<p align="justify">
$$
fib(n) =
\begin{cases}
	0, \quad n = 0 \\
	1, \quad n = 1\\
	fib(n-1) + fib(n-2), \quad n \geq 2
\end{cases}
$$

$$fib = \{ 0, 1, 1, 2, 3, 5, 8, \cdots \}$$

Complexity
$$T(0) = T(1) = 1$$

$$T(n) = T(n-1) + T(n-2) + 1, \quad n > 1$$

Let
$$S(n) = \frac{[T(n) + 1]}{2}$$

Then
$$S(0) = 1 = fib(1)$$

$$S(1) = 1 = fib(2)$$

So
$$S(n) = S(n-1) + S(n-2) = fib(n+1) = O(\Phi^{n}), \quad \Phi = \frac{1+\sqrt{5}}{2} = 1.61803$$

$$T(n) = 2S(n) - 1 = 2fib(n+1) - 1 = O(fib(n+1)) = O(\Phi^{n}) = O(2^{n})$$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_7_1_Fibonacci_1.png"/></center>
</p>
<p align="justify">
We can use dynamic programming to get fib(n)
</p>
{% highlight C++ %}
int fib(n)
{
	f = 0;
	g = 1;
	while (0 < n--)
	{
		g = g + f;
		f = g - f;
	}
	return g;
}
{% endhighlight %}

#### 1.7.2 Longest Common Sequence
<p align="justify">
Longest common sequence (LCS): two sequences have a common sequence (every iterm in orginal order) and this common sequence is the longest among all common sequences.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_7_2_Longest_Common_Sequence_1.png"/></center>
</p>
<p align="justify">
Suppose two sequences A[0, n] and B[0, m], we have two methods: recursion and iteration (dynamic programming).
<b>Recursion</b><br>
$$
LCS(A[0, n], B[0, m]) =
\begin{cases}
	'', \quad n = -1 || m = -1\\
	LCS(A[0, n), B[0, m)) + 'X', \quad A[n] = B[m] = 'X'\\
	max(LCS(A[0, n), B[0, m]), LCS(A[0, n], B[0, m))), \quad A[n] \neq B[m]
\end{cases}
$$
Worst condition is in $O(2^{n})$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_7_2_Longest_Common_Sequence_2.png"/></center>
</p>
{% highlight C++ %}
int getLCS(string s1, string s2)
{
    int n1 = int(s1.length()), n2 = int(s2.length());
    int **dp = new int *[n1+1];
    for (int i = 0; i < n1+1; i++)
    {
        dp[i] = new int [n2+1]{};
    }
    for (int i = 1; i < n1+1; i++)
    {
        for (int j = 1; j < n2+1; j++)
        {
            if (s1[i-1] == s2[j-1]) { dp[i][j] = dp[i-1][j-1] + 1; }
            else
            {
                dp[i][j] = (dp[i-1][j] > dp[i][j-1] ?
                            dp[i-1][j] : dp[i][j-1]);
            }
        }
    }
    int lcs = dp[n1][n2];
    for (int i = 0; i < n1+1; i++)
    {
        delete []dp[i];
    }
    delete []dp;
    return lcs;
}
{% endhighlight %}
<p align="justify">
<b>Iteration</b><br>
We construct a iterative table by n*m then initialize this table by first column and first row. Next, we fill out this table according to LCS equation.<br><br>

Worst condition is in $O(n*m)$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/1_7_2_Longest_Common_Sequence_3.png"/></center>
</p>

### 1.8 Abstract data type and data structure
<p align="justify">
ADT is one set of manipulations based on a data model while DS is an algorithm to implement the ADT with some language.
</p>


## 2. Vector
<p align="justify">
Vector is one ordered data structure. All data are linearly stored in memory. It supports call-by-rank, using O(1) time to determine one element's position.
$$A[0], A[1], A[2], \cdots, A[n-1]$$

Physical address:
$$V[ i ]=V[ 0 ]+i\times s$$

Where s denotes a space of each unit
</p>

### 2.1 Elementary Operations
<p align="justify">
According to one rule whether we can modify data structure, there are roughly 2 classes manipulation:<br>
(1) Satctic: read only, such as get, research<br>
(2) Dynamic: write, such as add, remove<br>
</p>

### 2.2 Extendable Vector
<p align="justify">
Load factor:
$$\lambda = \frac{size}{capacity}$$

We use extendable vector in order to avoid overflow and low load factor.<br>
<table class="c">
  <tr><th></th><th>Incremental capacity</th><th>Double capacity</th></tr>
  <tr><td>Total time</td><td>$O(n^{2})$</td><td>O(n)</td></tr>
  <tr><td>Amortized time</td><td>O(n)</td><td>O(1)</td></tr>
  <tr><td>Load factor</td><td>$\approx 100%$</td><td>> 50%</td></tr>
</table><br>
</p>

#### 2.2.1 Double Capacity
<p align="justify">
Each time we have a full vector, we expand it by multiplying 2
</p>
{% highlight C++ %}
void Vector<T>::expand()
{
	if (_size < _capacity) return
	_capacity = max(_capacity, DEFAULT_CAPACITY);
	T *oldElem = _elem;
	_elem = new T[_capacity <<= 1]; // Double
	for (int i = 0; i < _size; i++)
	{
		_elem[i] = oldElem[i];
	}
	delete []oldElem;
}
{% endhighlight %}
<p align="justify">
In worsed condition, we continue to insert n = $2^{m}$ >> 2 elements from an initial capacity of 1.<br><br>

We have to expand the vector when $1^{st}$, $2^{nd}$, $4^{th}$, $8^{th}$, $16^{th}$ (stating to insert by $0^{th}$)<br><br>

Total expand cost is
$$1, 2, 4, 8, \cdots, 2^m = n$$

Amortized cost is in $O(n)$
</p>

#### 2.2.2 Incremental Capacity
<p align="justify">
Each time we have a full vector, we expand it by adding a constant nuumber
</p>
{% highlight C++ %}
T *oldElem = _elem;
_elem = new T[_capacity += INCREMENT]; // Increment
{% endhighlight %}
<p align="justify">
In worsed condition, we continue to insert n = m*I >> 2 elements from an initial capacity of 1.<br><br>

We have to expand the vector when 1, $I+1$, $2I+1$, $3I+1$, \codts (stating to insert by $0^{th}$)<br><br>

Total expand cost is
$$0, I, 2I, 3I, \cdots, (m-1)I = I\frac{(m-1)m}{2}$$

Amortized cost is in $O(n)$
</p>

### 2.3 Unify
<p align="justify">
Deduplicate a batch of elements
</p>
{% highlight C++ %}
template <typename T>
int Vector<T>::uniquify()
{
	Ran i = 0, j = 0;
	while (++j < size)
	{
		if (_elem[i] != _elem[j]) _elem[++i] = _elem[j];
	}
	_size = ++i;
	shrink();
	return j-i;
}
{% endhighlight %}

### 2.4 Search
<p align="justify">
With respect to ordered vector
</p>

#### 2.4.1 Binary Search
<p align="justify">
Consider an order array A with n elements and one element e to find
$$
find(A[lo, hi], e) =
\begin{cases}
	mi, \quad A[mi = \frac{lo+hi}{2}] = e\\
	find(A[lo, mi), \quad A[mi] > e\\
	find(A(mi, hi], \quad A[mi] < e
\end{cases}
$$
</p>
{% highlight C++ %}
template <typename T>
static Rank binSearch(T *A, T const &e, Rank lo, Rank hi)
{
	while(lo < hi)
	{
		Rank mi = (lo+hi) >> 1;
		if (e < A[mi])
		{
			hi = mi;
		}
		else if (A[mi] < e)
		{
			lo = mi + 1;
		}
		else
		{
			return mi;
		}
	}
	return -1;
}
{% endhighlight %}
<p align="justify">
<b>Search length</b>: a number of comparison for keys. Binary search lenghth is in $O(1.5logn)$<br><br>

For example, at point 2, if we want to search 2, we have to do 2 comparison (check left and right). We can observe turn left and turn right have different comparison numbers.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/2_4_1_Binary_Search_1.png"/></center>
</p>
<p align="justify">
We use 2 comparison<br>
If e < A[mi], we enter left sub-vector; otherwise, we enter right sub-vector.
</p>
{% highlight C++ %}
template <typename T>
static Rank binSearch(T *A, T const &e, Rank lo, Rank hi)
{
	while(lo < hi)
	{
		Rank mi = (lo+hi) >> 1;
		(e < A[mi]) ? hi = mi: lo = mi+1;
	}
	return --lo;
}
{% endhighlight %}

#### 2.4.2 Fibonacci Search
<p align="justify">
Consider
$$n = fib(k) - 1$$

Let
$$mi = fib(k-1) - 1$$

1st sub-vector's length is $fib(k-1)-1$; 2nd sub-vector's length is $fib(k-2)-1$<br><br>

We can use this property to implement our search algorithm
</p>
{% highlight C++ %}
template <typename T>
static Rank fibSearch(T *A, T const &e, Rank lo, Rank hi)
{
	Fib fib(hi - lo); // Create a fib sequence in O(logn)
	while (lo < hi)
	{
		while (hi - lo < fib.get()) fib.preview();
		Rank mi = lo + fib.get() - 1;
		if (e < A[mi])
		{
			hi = mi;
		}
		else if (A[mi] < e)
		{
			lo = mi + 1;
		}
		else
		{
			return mi;
		}
	}
	return -1;
}
{% endhighlight %}
<p align="justify">
For any array A[0, n), we always select a cut point, $0 \leq \lambda < 1$.<br>
Binary search takes $\lambda = 0.5$, Fibonacci search takes $\lambda = \phi = 0.6180339$<br><br>

For an interval [0, 1), what is the best $\lambda$? Suppose average search length is $\alpha(\lambda)log_{2}n$. when $\alpha(\lambda)$ is smallest.
$$\alpha(\lambda)log_{2}n = \lambda [1 + \alpha(\lambda)log_{2}(\lambda n)] + (1 - \lambda)[2 + \alpha(\lambda)log_{2}((1 - \lambda)n)]$$

Derivative
$$\frac{-ln2}{\alpha(\lambda)} = \frac{\lambda ln\lambda + (1 - \lambda)ln(1 - \lambda)}{2 - \lambda}$$

When $\lambda = \phi$, $\alpha(\lambda) = 1.440420$ at minimum.
</p>

#### 2.4.3 Interpolation Search
<p align="justify">
We have a hypothesis: uniformly and independently distributed data.<br><br>

An ordered array A[lo, hi] have a linear trend
$$\frac{mi - lo}{hi - lo} \approx \frac{e - A[ lo ]}{A[ hi ] - A[ lo ]}$$

A possible cut point
$$mi \approx lo + (hi - lo)\frac{e - A[ lo ]}{A[ hi ] - A[ lo ]}$$

After one compariosn, n reduced to $\sqrt{n}$. Complexity is in $O(loglogn)$
</p>

#### 2.4.4 Search Strategy
<p align="justify">
Big scale: Interpolation search<br>
Medium scale: Binary search<br>
Small scale: Search in order
</p>


## 3. List
<p align="justify">
<b>List</b> is a classical structure by adopting dynamic storing whose elements are called <b>nodes</b>. Each node is connected by <b>point</b> to form a logically linear sequence, like
$$L = \left \{a_0, a_1, ..., a_{n-1} \right \}$$

Two adjacent nodes are called predecessor and successor for each other. First node which has no predecessor is called front or first. Similarly, the last node which has no successor is called rear or last.<br><br>

Differing from vector, if we want to visit some element, it is not suitable to call-by-rank for one list, because its elements are not linearly stored in memory. But one element is located by its neighbor. Therefore, it support <b>call-by-position</b>.<br><br>

<b>Why List cannot use Binary Search ?</b><br>
Because List cannont be efficiently accessed by rank.
</p>


## 4. Stack and Queue
### 4.1 Stack
<p align="justify">
A stack is open at its top, while other positions are not allowed to visit. LIFO (last in first out) is its speciality. In general, there are three basic operation for one stack: push, top and pop.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/ALGOS/DSA/4_1_0_1.png"/></center>
</p>
<p align="justify">
A stack can be implemented like Vetcor or List. Take Vector for instance, we can regard the head of vector is stack bottom and vetcor tail is stack top. Therefore, insert and remove is in O(1). In constrast, vector head is stack top makes insert ans remove in o(n).
</p>
{% highlight C++ %}
template <typename T>
class stack: public vector<T>
{
public:
	vois push(T const &e) {insert(size(), e);}
	T pop() {return remove(size() - 1);}
	T &top() {return (*this)[size()-1];}
};
{% endhighlight %}

#### 4.1.1 Base Convert
<p align="justify">
For example, convert decimal to binary
$$
89_{10} = 1011001_{2} \\
\\
\begin{aligned}
\frac{89}{2} & = 44, \quad & \text{mod} = 1 \\
\frac{44}{2} & = 22, \quad & \text{mod} = 0 \\
\frac{22}{2} & = 11, \quad & \text{mod} = 0 \\
\frac{11}{2} & = 5, \quad & \text{mod} = 1 \\
\frac{5}{2} & = 2, \quad & \text{mod} = 1 \\
\frac{2}{2} & = 1, \quad & \text{mod} = 0 \\
\frac{1}{2} & = 0, \quad & \text{mod} = 1
\end{aligned}
$$

From down to up, we write binary 1011001.
</p>
{% highlight C++ %}
void convert(stack<char> &S, _int64 n, int base)
{
    static char digit[] = {'0', '1', '2', '3', '4', '5',
        '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    while (n > 0)
    {
        S.push(digit[n%base]);
		n /= base;
    }
}
{% endhighlight %}

#### 4.1.2 Parenthesis
<p align="justify">
Parenthesis is useful for checking legal expressions. The expression above is illegal. We can take used of Stack to check parenthesis.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/parenthesis.png"/></center>
</p>
{% highlight c++ %}
bool paren(const char exp[], int lo, int hi)
{
	stack<char> S;
	for (int i = lo; i < hi; i++)
	{
		if ('(' == exp[i]) S.push(exp[i]);
		else if (!S.empty()) S.pop();
		else return false;
	}
	return S.empty();
}
{% endhighlight %}

#### 4.1.3 Stack Permutation
<p align="justify">
Consider A = < $a_{1}, a_{2}, \cdots, a_{n}]$, B = S = $\varnothing $<br><br>

Only two operations are allowed:<br>
(1) S.push(A.pop())<br>
(2) B.push(S.pop())<br><br>

After some legal operations, all elements in A are in B = [$a_{k1}, a_{k2}, \cdots, a_{kn}$><br>
We call such a process stack permutation<br><br>

If a stack has n elements, then how many possible stack permutations SP(n)?<br>
Suppose that we have a stack with n elements whose number is like 1, 2, 3,..., n, we take #1 element for instance, when #1 enter Stack B (at this time, Stack S is empty), we consider #1 as k-th element in Stack B. That is to say, before #1, there are k-1 elements and in Stack A, n-k elements are left. Well, the two groups are independent.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/stack_permutation_number.png"/></center>
$$SP(n)=\sum_{k=1}^{n}SP(k-1)\times SP(n-k)$$
$$SP(1)=1$$
</p>
<p align="justify">
We solve it
$$SP(n)=catalan(n)=\frac{(2n)!}{(n+1)!n!}$$

If we has a stack with three elements #1, #2, #3 (from top to bottom), then #3, #1, #2 is not a stack permutation. We can generalise, for any triple of elements i, j, k (relative position), k, i, j is not a stack permutation regardless of i, j, k is neighbor to each other or not.
</p>

#### 4.1.4 Infix Notation
<p align="justify">
We take two stacks for numbers and operators. When a new operator has lower or equal priority than top operator of stack operator, we calculate the top operator and we pop opertaor stack and push the new operator (if top operator in operator stack has low priority).
</p>
{% highlight C++ %}
float evaluate(char *S, char * &RPN)
{
	stack<float> opnd; // operator number
	stack<char> optr; // operator character
	optr.push('\0');
	while (!optr.empty())
	{
		if (isDigit(*S))
		{
			readNumber(S, opnd);
		}
		else
		{
			switch(orderBetween(optr.top(), *S)) {}
		}
	}
	return opnd.pop();
}
{% endhighlight %}

#### 4.1.5 Reverse Polish Notation(RPN)
<p align="justify">
RPN need not parenthesis. Compared to infix notation, RPN is logically simpler. For example, if we have an expression of 1+5, we can have it in format of RPN like 1 5 +.<br><br>

To implement RPN, we take use of a stack, each time we push number or operator into stack, when we encounter an operator, we pop two numbers in stack, calculate them and push the result into the stack. Continue this process until stack has only one element which is our final outcome.<br><br>

How to convert infix notation into RPN?<br>
Similar to infix notation compuation with stack, when we encounter an number, we append it to RPN; as for an operator, only if this operator cause a calculation, we append it to RPN, otherwise we push it into stack.<br><br>

For example: (0! + 1) ^ (2 * 3+ 4 - 5)<br>
(1) Show explicitely priority with '(' and ')'<br>
{ ([ 0! ] + 1) ^ ([ (2 * [ 3! ]) + 4 ] - 5) }<br><br>

(2) Move operator behind right brackets<br>
{ ([ 0 ]! 1) + ([ (2 [ 3 ]!) * 4 ] + 5) - } ^<br><br>

(3) Remove all brackets<br>
0 ! 1 + 2 3 ! * 4 + 5 - ^
</p>

### 4.2 Queue
<p align="justify">
A queue is a linear sequence and in its one side, only enqueue (at the tail of queue) is allowed, in constrast, in its other side, only dequeue (at the head of queue) is allowed. FIFO is fist in first out.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/queue_schema.png"/></center>
</p>
<p align="justify">
Queue has two limits:<br>
(1) Insert must be at the tail: enqueue() + rear()<br>
(2) Remove must be at the head: dequeue() + front()
</p>
{% highlight C++ %}
template <typename T>
class queue: public List<T>
{
	void enqueue( T const &e) {insertAsLast(e); }
	T dequeue() { return remove(first()); }
	T &front() { return first()->data; }
};
{% endhighlight %}

## 5. Tree
<p align="justify">
Tree is a special graph (V, E), number of nodes is |V|, number of edges is |E|. A tree is an undirected graph in which any two vertices are connected by exactly one path, or equivalently a connected acyclic undirected graph.<br><br>

If a tree has n nodes, it has n-1 egdes. For any node except root, it havs a unique path from root to itself. Thus, it has a unique path length (path length = number of edges along this path).<br><br>

Height of tree = max path length. Depth = Height + 1. If a tree has only one node, its height is 0. If a null tree, its height is -1.
</p>

### 5.1 Binary tree
<p align="justify">
If any node in tree has no more than 2 children, we define this tree as binary tree.<br><br>

In a tree, depth k level has 2^k nodes<br><br>

If a tree has n nodes and its height is h, h < n < 2^(h+1).<br><br>

Proper binary tree: a binary tree whose all node has a even out degree (0 or 2).<br>
Complete Binary Tree: Except last level, each level is complete, last level should be filled at left.<br>
Perfect Binary Tree: Each non-leaf node has two children, all leaf nodes are located in a same level.
</p>

### 5.2 Traverse
#### 5.2.1 PreOrder: V L R
<p align="justify">
Recursive
</p>
<p align="justify">
Interative
</p>
{% highlight C++ %}
template<class T>//Iterative pre order
void BST<T>::iterPreOrder(node<T>*curNode)
{
	stack<node<T>*> tree;
	tree.push(curNode);
	while (!tree.empty())
	{
		curNode = tree.top();
		tree.pop();
		BST<T>::visit(curNode);
		if (curNode->right != nullptr)
		{
			tree.push(curNode->right);
		}
		if (curNode->left != nullptr)
		{
			tree.push(curNode->left);
		}
	}
}
{% endhighlight %}

#### 5.2.2 InOrder: L V R
<p align="justify">
Recursive
</p>
<p align="justify">
Interative
</p>
{% highlight C++ %}
template<class T>//iterative in order
void BST<T>::iterInOrder(node<T> *curNode)
{
	stack<node<T>*> tree;
	while (curNode != nullptr)
	{
		while (curNode != nullptr)
		{
			if (curNode->right != nullptr)
			{
				tree.push(curNode->right);
			}
			tree.push(curNode);
			curNode = curNode->left;
		}
		curNode = tree.top();
		tree.pop();
		while (!tree.empty() && curNode->right == nullptr)
		{
			visit(curNode);
			curNode = tree.top();
			tree.pop();
		}
		visit(curNode);
		if (!tree.empty())
		{
			curNode = tree.top();
			tree.pop();
		}
		else
		{
			curNode = nullptr;
		}
	}
}
{% endhighlight %}

#### 5.2.3 PostOrder: L R V
<p align="justify">
Recursive
</p>
<p align="justify">
Interative
</p>
{% highlight C++ %}
template <class T>//iterative post order
void BST<T>::iterPostOrder(node<T>*curNode)
{
	stack<node<T>*> tree;
	node<T>*p, *q;
	p = q = curNode;
	while (p != nullptr)
	{
		for (; p->left != nullptr; p = p->left)
		{
			tree.push(p);
		}
		while (p->right == nullptr || p->right == q)
		{
			visit(p);
			q = p;
			if (tree.empty())
			{
				return;
			}
			p = tree.top();
			tree.pop();
		}
		tree.push(p);
		p = p->right;
	}
}
{% endhighlight %}

#### 5.2.4 LevelOrder
{% highlight C++ %}
template<class T>//level order
void BST<T>::levelOrder(node<T>*curNode)
{
	node<T> *last, *nextLast, *front;
	queue<node<T>*> queueTree;
	queueTree.push(curNode);
	last = nextLast = front = curNode;
	while (!queueTree.empty())
	{
		front = queueTree.front();
		queueTree.pop();
		BST<T>::visit(front);
		if (front->left != nullptr)
		{
			queueTree.push(front->left);
			nextLast = front->left;
		}
		if (front->right != nullptr)
		{
			queueTree.push(front->right);
			nextLast = front->right;
		}
		if (front == last)
		{
			cout << endl;
			last = nextLast;
		}
	}
}
{% endhighlight %}

### 5.3 Reconsturction of binary tree
<p align="justify">
PreOrder/PostOrder + InOrder -> binary tree
</p>


## 6. Binary seach tree
<p align="justify">
Any node is not less than its left child and not more than its right child. Obviously, BST' in-order is monotonously increasing.
</p>

### 6.1 How many possible BST with n nodes
<p align="justify">
If we have 4 nodes like <b>1, 2, 3, 4</b>, how many possible BST with them?<br>
Considerate n = 0 (Null tree), possible number of BST f(n) = 1<br>
n = 1, f(n) = 1<br>
n = 2, f(n) = 2<br>
n = 3, suppose k (k = 1, 2, 3) as root node, its left subtree has (k-1) nodes, its right subtree has (n-k) node. Total possible BST
$$f(n)=\sum_{k=1}^{n}f(k-1)f(n-k)=catalan(n)=\frac{(2n)!}{(n+1)!n!}$$
</p>

### 6.2 Self-balancing BST
<p align="justify">
In computer science, a self-balancing binary search tree is any node-based binary search tree that automatically keeps its height small in the face of arbitrary item insertions and deletions.<br><br>

For non self-balancing BST, we can rotate it to be self-balancing BST.<br>
Ideally balanced: h = log(n); Moderately balanced: h <= O(log(n))
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rotation_bst.png"/></center>
</p>

### 6.3 AVL
<p align="justify">
AVL: Adelson-Velsky & E. Landis<br>
They define an AVL tree by balance factor:<br>
bf(v) = height(v's left subtree) - height(v's right subtree)<br>
For any node, |bf(v)| = 0 or 1<br>
</p>

#### 6.3.1 AVL ~ BBST
<p align="justify">
An AVL tree with a height of h, has at least S(h) = fib(h+3)-1 nodes.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_example.png"/></center>
</p>
<p align="justify">
We can write a recursive equation for number of one AVL tree with a height of h. Like the example above, left tree is more high than right tree.
$$S(h)=1+S(h-1)+S(h-2)$$
$$S(h)+1=S(h-1)+1+S(h-2)+1$$

Make T(h) = S(h)+1
$$T(h)=T(h-1)+T(h-2)$$

This is a standard Fibonacci sequence (1, 1, 2, 3, 5,...). Now, T(h) corresponds to which term of Fibonacci sequence?<br>
If h = 0, AVL tree has only one node, T(0) = S(0)+1 = 2, which is 3rd term<br>
If h = 1, AVL tree has 2 or 3 node, T(1) = S(1)+1 = 3 or 4, which is 4th term. Here we take inferior border.<br>
So, we get a relation
$$T(h) = S(h)+1 >= Fib(h+3)=N$$

Fib(n) increases in an exponential way. So, the number of nodes has an inferior border.
$$N \sim \Omega(\Phi^{h})$$

Naturally, h has a superior border
$$h \sim O(log(n))$$

Thus, AVL satisfies the requirement of BBST(balanced BST): moderately balanced
</p>

#### 6.3.2 Unbalanced
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_unbalanced.png"/></center>
<p align="justify">
If we insert or remove one element in an AVL tree, it may be unbalanced. In details, insert an element, several subtress are unbalanced, for example, insert M, its ancestors K, N, R and G are unbalanced; while remove Y, only its parent R is unbalanced.
</p>

#### 6.3.3 Single rotation
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_signle_rotation_insert.png"/></center>
<p align="justify">
From the example, after inserting a node in the subtree of v, g becomes unbalanced, while v and p are not. Well, g is wanted for us, because it is the deepest node among all unbalanced nodes after inserting one node. We performe a signle rotation on g. A case of three generations g, p, v towards right is called zagzag, in contrast, zigzig. Attention, after this rotation, all ancestors of g return balanced if they were unbalanced before. Why?<br><br>

Before inserting one node, we have a base line (foot of T2 and T3), after inserting, our base line goes down 1 and this breaks balance rule. Besides, we notice a height for v, p, g increase by 1. After a single rotation, our base line comes back. We know, before inserting, our tree is balanced. Now our base line returns, ancestor of g is also balanced.<br><br>

A single rotation for inserting one node is in O(1).
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_single_rotation_remove.png"/></center>
</p>
<p align="justify">
As for removing one node, this is some different with inserting. From the example above, we remove one node of T3, this tree becomes unbalanced, we perform a zig on g. Finally, our tree return balanced. If T2 has a node, our base line comes back, but unlickily if T2 has no node, our base line goes up compared to that before removing, which potentially provoke an unbalanced of some ancestor of g.<br><br>

We have to check other some nodes if balanced or not. In the worst condition, we have to perform log(n) single rotations after removing one node.
</p>

#### 6.3.4 Double rotation
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_double_rotation_insert.png"/></center>
<p align="justify">
How to solve a case like the example above g's right child is p and p's left child is v (or symmetry case)? Answer is double rotation(zagzig or zigzag)
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/avl_double_rotation_remove.png"/></center>
</p>
<p align="justify">
As for removing one node in case of zagzig pr zigzag, our base line will goes up compared to that before removing, which means some ancestor will be unbalanced.
</p>

#### 6.3.5 3+4 reconstruction
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/34_reconstruct.png"/></center>
<p align="justify">
1) Suppose g is the deepest node among all unbalanced node, we care about three generations: g, p, v<br>
2) Rename them as a, b, c in the way of in-order<br>
3) For g, p, v, there is totally 4 subtrees (some subtree is possible null). Siminarly, we rename them as T0, T1, T2, T3 with method of in-order<br>
4) Because of monotonicity of in-order, we have a sequence like T0, a, T1, b, T2, c, T3<br>
5) Reconstruct a tree<br>
We can unify signle ration and double rotation with 3+4 reconstruction regardless of inserting or removing.
</p>

#### 6.3.6 Evaulation
<p align="justify">
pros:<br>
Insert, remove, search in o(log(n)). Stockage in o(n)<br>
cons:<br>
Rotation costs much for removing one node: o(log(n) in worst condition, 0.21 in average condition)
Topological space variation after single modification in Omega(log(n))
</p>


## 7. Advanced seach tree
<p align="justify">
AVL has a relatively rigorous reauirement even if it has been relaxed form ideal balance by balance factor. But AVL still need much cost to maintain its balance when we perform some insert or remove. So some advanced search trees play a necessary role.
</p>

### 7.1 Splay tree
#### 7.1.1 Locality and Self-adjusting
<p align="justify">
Locality: Some visited data just now is likely to be visited once more in a small time periode. Take BST as instance, it is likely to visit some data many times in a short time.<br><br>

Self-adjusting: consider a list, one element' visiting efficiency depends on its rank in the list, the smaller its rank is (or the nearer to head), the bigger vist efficiency is. If this list has a locality, we can do something on the list. For example, if one element in the list is visited, we put this element in the position of head. After a while, some frequently visited elements are aggregated in the front of list.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/splaytree_list_tree_1.png"/></center>
</p>

#### 7.1.2 Single splay
<p align="justify">
Naturally, we can generalize this concept to a tree or BST. Concretely, list's head corresponds to tree's top anf list's tail correspons to tree's bottom.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/splaytree_list_tree_2.png"/></center>
</p>

<p align="justify">
So, we have a strategy: once one element is visited, we move it to the root of the tree by zig or zag. Because one zig or zag can level up an element once. We repeat zig or zag until the element is moved to root node.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/splaytree_zigzag.png"/></center>
</p>

<p align="justify">
We talk about the worst condition: we have a tree like this. We hope to search each element in a cycle.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/splaytree_worst_condition_1.png"/></center>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/splaytree_worst_condition_2.png"/></center>
</p>
<p align="justify">
We can find out a whole cycle is in $\Omega(n^{2})$ and amortized complexity is in $\Omega(n)$. This is much bigger than log(n). So, we have to do some change.
</p>

#### 7.1.3 Double splay
<p align="justify">
Double splay: for three generations g = parent(p), p = parent(v), v, lift v two levels to be root with at most 2 rotations.<br>
For zigzag (or zagzig), double splay has no difference avec single AVL and single splay: we have to zig on p then zag on g to left v two levels.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/double_splay_zigzag.png"/></center>
</p>
<p align="justify">
But for zigzig (or zagzag), double splay is different: zig on g then zig on p to lift v two levels while single splay zig on p then zig on g. According to their final result, these two splay manage to lift v two levels, but they have a different effect on the sub tree.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/double_splay_zigzig.png"/></center>
</p>
<p align="justify">
To clarify the difference between single splay and double splay on zagzag (or zigzig). We try to visit the deepest node.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/double_splay_zagzag_example.png"/></center>
</p>
<p align="justify">
The power of double splay is to reduce subtree's height in an exponential way, while single splay does the same thing in an arithmetic progression. amortized complexity for double splay is $O(log(n)))$.<br><br>

Obviously, there is a special case: v has no grandparent. In this case, v's parent is root for the whole tree. Then, one zig or zag (it depends on concrete morphology) is enough.<br><br>

Evaluation:<br>
No need to record height of node and balance factor, it is easier to implement than AVL<br>
amortized complexity is $O(log(n))$ which is equal to AVL<br>
If some locality exists, more efficient to visit cache data. In other word, during an interval of time, we visit some data more frequently. If we have n data, k data is much frequently visited, m is visiting times, so k << n << m. Then searching m times is in $O(mlog(k)+nlog(n))$.<br>
But splay tree cannot guarantee advoiding the worst condition and splay tree is not suitable for some occasion which is sensible for efficiency.
</p>

### 7.2 B-tree
<p align="justify">
640 KB ought to be enough for anybody. -- B. Gates, 1981<br><br>

Two facts:<br>
1 s = 1 day: if we visit RAM with 1 s, we visit a disk with 1 day.<br>
1 B = 1 KB: time of visiting 1B is nearly equal to that of visiting 1KB.
</p>

#### 7.2.1 Definition
<p align="justify">
B-tree (R. Bayer & E. McCreight) is a mutil-way tree. All external nodes (children of leaf node but null) have a same depth. The height of B-tree is from root to external nodes instead of leaf node. Each node is called super node.<br>
Merge 2 generations as a super node: 3 key values and 4 branches<br>
Merge d generations as a super node: $2^{d}-1$ key values and $m=2^{d}$ branches<br>
If there are N internal keys values in this B-tree, there are N+1 external nodes (key value is null)<br><br>

In fact, B-tree is logically equal to BBST. But, compared to BBST, B-tree visit a batch of values at each super node in order to reduce I/O, because each level represents one I/O.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_merge.png"/></center>
</p>
<p align="justify">
How many internal nodes in a m-way B-tree?<br>
At most m-1 key values and m branches; at least $\left \lceil \frac{m}{2} \right \rceil$ branches except root node with 2. Besides, we can call some B-tree according to its superior limit and inferior limit, e.g. we call 4-way B-tree (2, 4) tree and 7-way B-tree (4, 7) tree.<br><br>
</p>

#### 7.2.2 Height
<p align="justify">
If we have a m-way B-tree with N key value, what is it's maximun height?<br>
At each level, internal node should be as less as possible. $n_{0}=1$, $n_{1}=2$, $n_{2}=2\times \left \lceil \frac{m}{2} \right \rceil$<br>
For $k^{th}$ level ($k\geq 1$)
$$n_{k}=2\times \left \lceil \frac{m}{2} \right \rceil^{k-1}$$

For last level in which all external node are. Because there are N internal key value, external key values are N+1.
$$N+1=n_{h}\geq 2\times \left \lceil \frac{m}{2} \right \rceil^{h-1}$$

$$h\leq 1+log_{\left \lceil \frac{m}{2} \right \rceil}\left \lfloor \frac{N+1}{2} \right \rfloor=O(log_{m}N)$$

Compared to BBST
$$\frac{log_{\left \lceil \frac{m}{2} \right \rceil}\frac{N}{2}}{log_{2}N}=\frac{1}{log_{2}m-1}$$

A 256-way B-tree has a $\frac{1}{7}$ of BBST's height (I/O times)<br><br>

How about its minimum height?<br>
Each internal node has as many key values as possible. So, $n_{0}=1$, $n_{1}=m$, $n_{2} = m^{2}$, ..., $n_{h-1}=m^{h-1}$, $n_{h}=m^{h}$.<br>
For the last level in which all external node are
$$N+1=n_{h}\leq m^{h}$$

$$h\geq log_{m}(N+1)=\Omega(log_{m}N)$$

Compared to BBST
$$\frac{log_{m}N-1}{log_{2}N}=log_{m}2-log_{N}2=\frac{1}{log_{2}m}$$

A 256-way B-tree has a $\frac{1}{8}$ of BBST's height. In fact, a m-way B-tree has a relatively fixed height.
</p>

#### 7.2.3 Insert
<p align="justify">
If we insert a new key value in a super node and an overflow happens, we have to split this node. In fact, this super node has already m key values such as $k_{0}, k_{1}, ..., k_{m-1}$. We take its median $s=\left \lfloor \frac{m}{2} \right \rfloor$ and split all key values with $k_{s}$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_overflow_1.png"/></center>
</p>
<p align="justify">
Besides, we lift $k_{s}$ one level to join its parent and set the rest 2 parts as left children and right children of $k_{s}$. For example, after inserting 37, overflow happens, we put 37 into its parent and set 17, 20, 31 as its left children and 41, 56 as its right children.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_overflow_2.png"/></center>
</p>
<p align="justify">
Possibly, this kind of overflow will pass upward until root. We split this super node and establish a new root and take $k_{s}$ a key value in this new root. For example, an overflow in root node, we establish a new node with only one key value 37, and set 17, 20, 31 as its left children and 41, 56 as its right children. At this time, height adds 1 and number of branches for the new root is 2.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_overflow_3.png"/></center>
</p>
<p align="justify">
Insert is in $O(h)$<br><br>
</p>

#### 7.2.4 Remove
<p align="justify">
As for removing, if some key value to be removed is not a leaf, we find its most left children in its right subtree and exchange their position, then remove the key value. Similarly, remove may provoke underflow.<br>
If some super node has an underflow after removing one key value, at this time it has $\left \lceil \frac{m}{2} \right \rceil-2$ key values and $\left \lceil \frac{m}{2} \right \rceil-1$ branches. There are two methods for solving underflow.<br>
<b>Rotation</b>: if this super node v's sibling s has more than $\left \lceil \frac{m}{2} \right \rceil-1$ key value, v can borrow a key value from s with respecting the in-order rule. For example, y move to first one of v, x replace y's orginal position.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_underflow_1.png"/></center>
</p>
<p align="justify">
What if v's lfet sibling and right sibling have no available key value?<br>
<b>Combine</b>: Suppose v's left sibling is s, s has exactly $\left \lceil \frac{m}{2} \right \rceil-1$ key values. We combine s, v and their parent as a new super node with a number of 
$$\left \lceil \frac{m}{2} \right \rceil-1+\left \lceil \frac{m}{2} \right \rceil-2+1<m-1$$

At the same time, combine y's left pointer and right pointer together at the new super node. Naturally, this process will pass upward until root node.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/b_tree_underflow_2.png"/></center>
</p>
<p align="justify">
Remove is in $O(h)$<br><br>
</p>

### 7.3 Red-Black tree
<p align="justify">
Ephemeral data structure: some state exists in one moment. If we have a dynamic change on some data structure such as list, stack or graph, its state will change without perserving its ancient state.<br><br>

Persistent structure: support a visit for ancient state. How to realize this structure? A sample method is to preserve its each history version, but this way will cause much waste for some data because of duplicated copy.<br><br>

Take a BBST as an instance, we preserve all history version (h = |history|). Each search is in $O(logh+logn)$, because we have to locate version then search some element. Totally, time/space complexity is in $O(h+n)$.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_1.png"/></center>
</p>
<p align="justify">
In fact, we update a few elements between two neighbor versions and most elements are not updated. This is a relationship. So we just maintain some updated elements. Then time complexity for sigle version (or augmentation) is in $O(logn)$ and total complexity is in $O(n+h)$.<br><br>

We want reduce time complexity to $O(n+h)$. So we have to require a topological difference in $O(1)$ between 2 neighbor versions. Unluckily, most BBST cannot guarantee this requirement. For example, we have a dynamic operation on an AVL such as insert or remove. In order to keep its balance, we will do some rotations. It's ok for inserting because of at most 2 rotations, but the worst condition for removing is in $O(logn)$. Therefore, we need a specific tree to satisfy our requirement (topological structure variation in $O(1)$). This is Red-Black tree.
</p>

#### 7.3.1 Definition
<p align="justify">
From the name of Red-Black tree, it has two kind of colors: red and black. We supply external nodes for each leaf node of Red-Bed tree so that it's a true BST(each node has two children even if its children are null).<br><br>

A Red-Black tree can be defined as follows:<br>
1) root must be black<br>
2) all external nodes must be black<br>
3) for other nodes, if they are red, their children must be black (red's children and parent are black)<br>
4) from root to any external node, the number of black nodes passed must be identic.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_2.png"/></center>
</p>

#### 7.3.2 Lifting
<p align="justify">
In order to better understand Red-Black tree. We need a technique -- lifting. Concretely, we lift each red node one level so that it has a same height as its parent.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_lift_1.png"/></center>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_lift_2.png"/></center>
</p>
<p align="justify">
We can observe that all nodes in the last level has a same height after lifting. In fact, this is a (2, 4) B-tree now. Besides, a Red-Black tree is balanced because B-tree is balanced. If there is n internal nodes in a Red-Black tree, its height is $h = O(logn)$<br><br>

Black height (bh): a number of balck node in a path from root to any external node. According to Red-Black tree's property, a red node's child and parent must be black, which means a Red-Black tree's height is between one black height and 2 black height. How to compute its black height? Answer is its equivalent B-tree's height.
$$bh \leqslant 1+log_{\left \lceil \frac{m}{2} \right \rceil} \frac{n+1}{2} \leqslant log_{2} (n+1)$$

A Red-Black tree's height
$$bh \leq h \leq 2\times bh$$
$$log_{2}(n+1) \leq h \leq 2\times log_{2}(n+1)$$
</p>

#### 7.3.3 Insert
<p align="justify">
Algorithm:<br>
Suppose a node named x to be inserted, according to BBST's insert algorithm, we insert x as a leaf node with two external children and color x in red. At this time, definition 1), 2), 4) are satistied but 3) is not for sure. Because there is a risk x's parent p is red. We call this case double red. If p is black, everything is ok, otherwise, we have to get rid of double red.<br><br>

In addition, we are sure g is black if p is red but we are not sure g's anothr child u is red or black.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_1.png"/></center>
</p>
<p align="justify">
According to u's color, we have 2 situations.<br><br>

<b>If u is black</b>, p and x is red, we consider three generations g, p, x. Naturally, there are 4 kinds of topological structure: zagzag, zigzig, zagzig and zigzag. Here, we only take zigzig and zagzig into account because of symmetry. In this case, u may be an external node or a leave node, but this tree's black doesn't change. We lift all red node one level to form a B-tree (4-way is legal) then we can observe a super node with x, p, g or p, x, g. No matter whcih condition, two red node are adjacent. What we should do is reorginize this tree value so that red-black-red. For zigzig, exchange p and g's color; for zagzig, exchange x and g's color.<br><br>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_2.png"/></center>
</p>
<p align="justify">
Another way is 3+4 reconstruction, we sort a, b, c as well as their subtree (if not null) to reconstruct a BBST and color a, c in red and the others in black. This method can handle all topological structures.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_3.png"/></center>
</p>
<p align="justify">
Only one recolor is enough to get rid of double color and the Red-Black tree's topological structure doesn't change, so complexity for this situation is in $O(1)$.<br><br>

<b>If u is red</b>, similarly we take three generations g, p, x into account for zigzig and zagzig. We lift red nodes, then we can observe the super node has 4 key values with an overflow happened. So, we need to split the super node to eliminate overflow. Equivalently, in the view of Red-Black tree, p and u change to black, g changes to red. Take care that g goes up one level (or g gets red) may cause a new double red, so we can continue our algorithm until no double red. Although, double red will pass upward and total recolor may be in $O(logn)$, our topological structure of Red-Black tree doesn't change. During this process, what we only do is change color instead of rotations. So, topological structure variation is in $O(1)$.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_4.png"/></center>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_5.png"/></center>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_insert_6.png"/></center>
</p>
<p align="justify">
In other word, reconstruction is in $O(1)$. Why do we focus on reconstruction more than coloring? Because reconstruction of tree has a relationship with persistent structure. If some sturcture os persistent, the less reconstruction happens, the better.
</p>

#### 7.3.4 Remove
<p align="justify">
Algorithm:<br>
Suppose we want to remove x, first of all we have to find x's successor r. According to normal BST's algorithm, we usually find the leftmost child in x's right subtree. Attention, we ignore external nodes. If one of x and r is red, everything is ok, otherwise, we have a double black (double red is impossible).
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_1.png"/></center>
</p>
<p align="justify">
If both x and r are black, after removing x, we reduce our black height by 1, so rule 4 is not satisfied. Besides, we call x's parent p and p's another child s (x's sibling). We have four cases.<br><br>

<b>Case 1: if s is black and has at least one red child t</b>
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_2.png"/></center>
</p>
<p align="justify">
We take use of B-tree to elaborate why it works. In fact, after remvoing x, an underflow happens in the supernode of x. Lucily, x can borrow a key value form its sibling s by rotating.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_3.png"/></center>
</p>
<p align="justify">
<b>Case 2: if s is black with two black children and p is red</b><br>
x/r keeps balck, s changes to be red and p changes to be black.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_4.png"/></center>
</p>
<p align="justify">
Is an underflow possible to propagate upward? No, because p is red so p's parent must be black. In the viwe of B-tree, there must be a value with p together in some super node. Although p goes down to combine with its children, the original super node couldn't have an underdlow.<br><br>

<b>Case 3: if s is black with two black children and p is black</b><br>
s changes to be red.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_5.png"/></center>
</p>
<p align="justify">
In the view of B-tree, un underflow continues to propagate upwards so such a combination may be in $O(logn)$. However, in the view of Red-Black tree, there is no strcuctural variation. What we need do is color s in red.<br><br>

<b>Case 4: if s is red (obviously with 2 black children or null)</b><br>
zig(p) (or zag(p)), s changes to be black and p changes to be red.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_6.png"/></center>
</p>
<p align="justify">
After such an operation, we do not solve double black and rule 4 is still unsatisfied. But we transform case 4 to case 1 or case 2. Then we do the same thing like case 1 or case 2 to get rid of double black.<br><br>

In a word, remove a node in $O(logn)$, with at most $O(logn)$ coloring, one 3+4 reconstruction (rotation), one single rotation (zig or zag).
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/rb_tree_remove_7.png"/></center>
</p>


## 8. Graph
### 8.1 Basic concept
<p align="justify">
G = (V, E)<br>
vertex: n = |V|, edge or arc: e = |E|<br>
adjacency: v ~ v<br>
incidence: v ~ e<br>
directed vs. undirected<br>
path vs. cycle<br>
DAG: directed acylic graph<br>
Eulerian path/cycle: pass all edges once<br>
Hamiltonian path/cycle: pass all vertex once<br>
adjaceny matrix: n×n; incidence matrix: n×e
</p>

### 8.2 Searching
#### 8.2.1 Breath-first search
<p align="justify">
1) Discover first vertex v, discover its neignbor vertex, change v' status to be visited<br>
2) For each discovered neighbor, discover its neighbor, change its status to be visited<br>
3) Repeat util all vertex are visited<br>
</p>

#### 8.2.2 Depth-first search
<p align="justify">
1) Start from first vertex, pick up its one neighbor (randomly)<br>
2) For current vertex, do same thing as first vertex<br>
3) If on avaiable vertex, backtracking
</p>


## 9. Dictionary
### 9.1 Hash
<p align="justify">
Call-by-rank: Vector<br>
Call-by-position: List<br>
Call-by-key: BST<br>
Call-by-value: Hashing<br><br>

Consider we want to visit some value with a key, an usual way is to use a vector to contain all value. It os easy to call-by-rank. Luckily, this will occupy much ressource. We hope to make use of a small vertor to realize this function. So we introduce hash table (or bucket array) with a lenghth (capacity) of M.<br><br>

Our hash table must be capable of accomodating all value N. At the same time, hash table is much smaller than real key space
$$N < M << R$$
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/hashing_1.png"/></center>
</p>
<p align="justify">
Concretely, we want to search a phone number with 8-digits and we know there is totally 90000 phone numbers. How do we design our searching system? Simply. we can set a vector to contain all possible numbers, say $10^{8}$. Obviously, this occupies much ressources and has a low use rate, because we search 90000 phone numbers but have to spend so much memory (use rate = $\frac{90000}{10^{8}}$). So, we turn to hash table. We set a hash table with a length of 90000 and a hash function
$$hash(key) = key \% 900001$$

If we want to search 62785001, we calculate its remainder by 90001. Then, we can call-by-rank in hash table to acquire the phone number. Apparently, hash highly augment a use rate<br><br>

But, what if two keys (e.g. 51531876 and 62782001) have a same result by hash table? Such a problem is called hash collision.
</p>

### 9.2 Hash function
<p align="justify">
Hash function is map from a key space to hash table, namely hash(): S $\rightarrow $ A. But key value's amount is much bigger than hash table's capacity $\left | S \right | = R >> M = \left| A \right|$, so hash function isn't a single map.<br><br>

What does a good hash function look like?<br>
1) determinism: a same key is always mapped to a same address<br>
2) efficiency: expect $O(1)$<br>
3) surjection: all keys are distributed in the hash table as more as possible<br>
4) uniformity: a uniform probability of mapping one key to hash table in order to avoid clustering<br><br>
</p>

#### 9.2.1 Modulo operation
<p align="justify">
$$hash(key) = key \% M$$

It would be better that M is a prime number. Why? Consider we have a hash table with a length of M, now here is a key S. We focus at their greatest common divisor
$$gcd(S, M) = g$$

We hope g can be uniformly distributed in our hash table for any s. According to number theory, g should be 1. So, M is a prime. 
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/hashing_2.png"/></center>
</p>
<p align="justify">
Modulo operation is not perfect, because 0 is always mapped to 0 no matter what M is ($hash(0) \equiv 0$). Besides, if s and s' is near to each other, thier maps are still near. These will break uniformity.
</p>

#### 9.2.2 MAD
<p align="justify">
MAD is multiply-add-divide. We define M by a prime number, with a > 0, b > 0 and a % M $\neq$ 0.
$$hash(key) = (a \times key + b) \% M$$

b is offset to avoid fixed point and a is step to disperse some near points.
</p>

#### 9.2.3 Selecting digits
<p align="justify">
We take some digits (decimal or binary) for a key. For example, we take all odd digits position 1, 3, 5,.. for a decimal number
$$hash(123456) = 246$$
</p>

#### 9.2.4 Mid-square
<p align="justify">
We take some digits in the middle of $key^{2}$. For example, $hash(123) = 15129 = 512$
</p>

#### 9.2.5 Folding
<p align="justify">
We split a key into several parts then sum them. For example,
$$hash(123456789) = 123 + 456 + 789 = 1368$$
</p>

#### 9.2.6 XOR
<p align="justify">
We split a key into several parts with a same width, then have a XOR operation on them. For example,
$$hash(110011011) = 110 \wedge 011 \wedge 011 = 110$$
</p>

#### 9.2.7 Pseudorandom number generator
<p align="justify">
$$hash(key) = rand(key) = (rand(0) \times a^{key}) \% M$$

Different platforms may use different algorithm to generate a random number. So, we shoul take care of compatibility.
</p>

#### 9.2.8 Polynomial
<p align="justify">
$$hash(s = [x_{0}, x_{1}, ..., x_{n-1}]) = x_{0}a^{n-1} + x_{1}a^{n-2} + ... + x_{n-2}a^{1} + x_{n-1}$$

Here is an approximate way that we remplace all multiplication by base convert
</p>
{% highlight C++ %}
static size_t hashCode(char s[])
{
	int h = 0;
	for (size_t n = strlen(s), i = 0; i < n; i++)
	{
		h = (h << 5) | (h >> 27);
		h += (int)s[i];
	}
	return (size_t) h;
}
{% endhighlight %}


### 9.3 Collision resolution
#### 9.3.1 Open hashing
<p align="justify">
For any collision address, we open a chain for them.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/hashing_3.png"/></center>
</p>
<p align="justify">
Pros:<br>
No need to provide in advance some space for any node.<br>
Support endless collision if system permits.<br><br>

Cons:<br>
Extra space for pointer<br>
Dynamic allocation for new node<br>
System cache doesn't work
</p>

#### 9.3.2 Open addressing or Closed hashing
<p align="justify">
Not like open hashing, we prepare in advance several space buckets for all collisions forming a probing sequence/chain. If we want to seach some value, we visit along this probing sequence until we find this value or a empty bucket. How to establish the probing sequence?<br><br>

<b>Linear probing</b><br>
Take modulo for an insatnce, once a collision happens, we turn to its successor
$$\begin{matrix} [hash(key) + 1] \% M \\ [hash(key) + 2] \% M \\ [hash(key) + 3] \% M \\ \vdots \end{matrix}$$

Until we succeed in find it or fail.<br><br>

Closed hashing efficiently take use of current space of hash table instead of allocating new space. Besides, hash table is still continuous. However, close hashing may cause a manipulation time > $O(1)$ and provoke a sequential collision because of occupation.<br><br>

<b>Lazy removal</b><br>
If we want to remove some key in a bucket, we have to leave a remark on this bucket after remving this key in order to avoid losing all sequential keys. Because we wil mistake this bucket is empty if no remark is left.<br><br>

<b>Quadratic probing</b><br>
We prob next bucket in a quadratic number<br>
$$\begin{matrix} [hash(key) + 1^{2}] \% M \\ [hash(key) + 2^{2}] \% M \\ [hash(key) + 3^{2}] \% M \\ \vdots \end{matrix}$$

Quadratic probing is helpful to avoid collision cluster, mais may potentially increase I/O. Besides, here is another question, is it possible that some bucket is never visited by quadratic probing<br><br>

Suppose we have hash time with a lenghth M = 11. We can calculate all poosible bucket number by quadratic probing.
$$\{0, 1, 2, 3, 4, 5,...\}^{2} \% 11 = \{0, 1, 4, 9, 5, 3\}$$

We can observe first 6 ($\left \lceil \frac{11}{2} \right \rceil$) keys map to different address to each other and about 50% buckets are never visited.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/hashing_4.png"/></center>
</p>
<p align="justify">
<b>Proof by contradiction</b>: any two keys less than $\left \lceil \frac{M}{2} \right \rceil$ are different to each other.<br>
Assume $\exists$ a, b
$$0\leq a \lt b \lt \left \lceil \frac{M}{2} \right \rceil$$

We know $a^{th}$ and $b^{th}$ are in collision by duadratic probing.<br>
So
$$a^{2} \% M = b^{2} \% M$$

Then
$$(b^{2}-a^{2}) \% M = (b-a)(b+a) \% M = 0$$

According to number theory
$$0 \lt b-a \lt b+a < M$$

This is contrast with our hypothesis M is prime. Therefore, any two keys less than $\left \lceil \frac{M}{2} \right \rceil$ are different to each other.<br><br>

<b>Quadratic forward and backward probing</b><br>
We can use 2 probing, one is forward and the other is backward.
$$\begin{matrix} [hash(key) + 1^{2}] \% M\\ [hash(key) - 1^{2}] \% M\\ [hash(key) + 2^{2}] \% M\\ [hash(key) - 2^{2}] \% M\\ [hash(key) + 3^{2}] \% M\\ [hash(key) - 3^{2}] \% M\\ \vdots \end{matrix}$$

But for some prime number like 5, 13, forward chain and backward chain are compose of a same group of numbers, while other prime number like 7, 11 are not.
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/DSA/hashing_5.png"/></center>
</p>
<p align="justify">
How to chose prime number M? It would be better $M = 4\times k + 3$, like 7, 11, 19, 23, ..... Why?<br><br>

Two-square Theorem of Fermat:<br>
Any prime number p can be equal to a sum of two integer's square, if and only if p % 4 = 1<br><br>

$$(u^{2}+v^{2})(s^{2}+t^{2}) = (us+vt)^{2}+(ut-vs)^{2}$$

$$(2^{2}+3^{2})(5^{2}+8^{2}) = (2\times5+3\times8)^{2}+(2\times8-3\times5)^{2}$$

We can infer, any natural number n can be equal to a sum of two integer's square if and only if each n's prime decomposition factors like $M = 4k + 3$ has a even power. For example
$$810 = 2 \times 3^{4} \times 5$$

810 has 3 prime factors: 2, 3, 5 where only 3 is like $M = 4k + 3$. At this time, 3 has a power of 4. Therefore, $810 = 27^{2} + 9^{2}$<br><br>

Now we have a prime number M = 4k + 3, assume there exist two different numbers a, b $\in [1, \left \lfloor \frac{M}{2} \right \rfloor]$ and $a^{2}$, $-b^{2}$ have a collision.
$$a^{2} \% M = -b^{2} \% M$$

$$(a^{2} + b^{2}) \% M = 0$$

According to the inference above, $n = a^{2} + b^{2}$ has a prime factor M with a format of 4k + 3, M's power is even, namely at least 2. In other word,
$$n \% M^{2} = 0$$

$$a^{2} + b^{2} \geq M^{2}$$

But this is impossible, so there is no such two integers a, b. In other word, forward and backward probing have no common bucket.
</p>


## 10. Priority queue
<p align="justify">
Call-by-priority: we hope to visit certain element with a highest priority.
</p>

### 10.1 Complete Binary Heap 
<p align="justify">
Complete Binary Tree is special AVl with a non-negative balance factor everywhere. In other word, left subtree is not short than right subtree.<br><br>

Logically: complete binary heap is equal to a complete binary tree<br>
Physically: complete binary heap is equal to a vector<br><br>

If an element's rank is i, its parent (if exists) has a rank of $\frac{i-1}{2}$, its left child (if exists) has a rank of 2i+1, its right child (if exists) has a rank of 2i+2.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/complete_binary_tree_1.png"/></center>
</p>

### 10.2 Max heap
<p align="justify">
We define a max heap by $H[i] \leq H[parent(i)]$ for any i > 0. We call this rule heap order. So, the max value must be at root node for a max heap. Similarly, we can define a min heap.
</p>

### 10.3 Insert
<p align="justify">
For any new element e, we insert it at the end of vector. If e $\leq$ parent(e), everything is ok; otherwise, we exchange e and its parent. However, it is possible that e is still bigger than its new parent, so we continue to exchange e and its parent untill heap order is satisfied. Such a method for adjusting is called percolate up.<br><br>

In fact, each we don't need 3 assignments at each swap. We can get a copy of e, until we find some a parent that heap order is satisfied we swap e and this parent.<br><br>

During this process, the complexity for assignment is in $O(logn+2)$.
</p>

### 10.4 DelMax
<p align="justify">
Complete binary heap only support visit the max and remove the max. If we want remove the max e, we exchange e and the last node r then remove e. Obviously, new root r may be samller than its children. In this case, we swap r and r's bigger child between two child. We continue this operation until heap order is respected. We call this process percolate down. Similarly, we can find the last child c and swap c and r.<br><br>

Complexity is in $O(logn)$
</p>

### 10.5 Heapification
<p align="justify">
We have n elements in an array and we want build a heap with them.
</p>

#### 10.5.1 Top-to-Down percolate up
<p align="justify">
We insert all elements at the end of heap by the first one then percolate up to adjust the heap.
</p>
{% highlight C++ %}
void percolateUp(vector<int> &arr, int idx)
{
    int n = int(arr.size());
    if (idx < 0 || idx >= n) { return; }
    while (idx > 0)
    {
        int parIdx = (idx - 1) / 2;
        if (arr[parIdx] > arr[idx])
        {
            swap(arr[parIdx], arr[idx]);
            idx = parIdx;
        }
        else { return; }
    }
}

void heapify(vector<int> &arr)
{
    int n = int(arr.size());
    for (int i = 1; i < n; i++) { percolateUp(arr, i); }
}
{% endhighlight %}
<p align="justify">
For worst condition, complexity is in $O(nlogn)$. In fact, we can sort them with such time. So, this way don't saisfy our demand.
</p>

#### 10.5.2 Down-to-Up percolate down
<p align="justify">
Consider a case: we have two sub-heap $r_{0}$ and $r_{1}$ which are children of p. How to adjust the heap?
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/heapification_1.png"/></center>
</p>
<p align="justify">
Answer is percolate dowm p. Besides, we know there oare more than 50% nodes are all leaf nodes. So, we only need do this operation for internal nodes. Last internal node's position is $\left \lfloor \frac{n}{2} \right \rfloor - 1$
</p>
{% highlight C++ %}
void percolateDown(vector<int> &arr, int idx)
{
    int n = int(arr.size());
    if (idx < 0 || idx >= n) { return; }
    while (idx < n)
    {
        int leftIdx = idx * 2 + 1, rightIdx = idx * 2 + 2;
        if (leftIdx >= n) { break; }
        if (rightIdx >= n) { rightIdx = leftIdx; }
        if (arr[idx] < min(arr[leftIdx], arr[rightIdx])) { return; }
        if (arr[leftIdx] < arr[rightIdx])
        {
            swap(arr[idx], arr[leftIdx]);
            idx = leftIdx;
        }
        else
        {
            swap(arr[idx], arr[rightIdx]);
            idx = rightIdx;
        }
    }
}

void heapify(vector<int> &arr)
{
    int n = int(arr.size());
    for (int i = n-1; i >= 0; i--) { percolateDown(arr, i); }
}
{% endhighlight %}
<p align="justify">
For worst condition, complexity is in $\sum_{i} height(i) = O(n)$.
</p>

### 10.6 Leftist Heap
<p align="justify">
Leftist heap is to merge two heaps A and B efficiently.<br><br>

A simple way is A.insert(B.removeMax) in $O(m*(logm+log(n+m))) = O(m*log(m+n))$<br><br>

Another way by Floid Heapification is union(A, B) then heapify(A+B) in $O(n+m)$<br><br>

We seek for a faster way -- Leftist Heap in $O(logn)$<br>
Nodes are inclined to be located at left and nodes to be merge at right.
</p>

#### 10.6.1 Null Path Length
<p align="justify">
We introduce external nodes as null nodes. We define null path length (NPL):<br>
$$
NPL(x) =
\begin{cases}
	0, \quad x \text{is null} \\
	1 + min(NPL(lc(x)), MPL(rc(x))), \quad \text{otherwise}
\end{cases}
$$

Where lc(x) is x's left child adn rc(x) is x's right child<br><br>

In fact, if we change min with max, we will get a height of tree.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/NPL_1.png"/></center>
</p>
<p align="justify">
For example, NPL for node 3 is 3.<br><br>

NPL(x) means a minimum distance form x to some external node.<br>
NPL(x) means a tree with x as root has a complete binary tree of a height of NPL(x)
</p>

#### 10.6.2 Leftist Property
<p align="justify">
For any node, NPL(lc(x)) $geq$ NPL(rc(x)), we call this leftist property.
$$NPL(x) = 1 + NPL(rc(x))$$

A heap with leftist property is a leftist heap. We can infer that a leftist heap's sub heap is a leftist heap.
</p>

#### 10.6.3 Right Chain
<p align="justify">
Starting from x, go down always along right sub heap until an external node. We call this path right chain. Obviously, the end of a right chain must be an external node with the smallest NPL.
$$NPL(r) = \left | rChain(r) \right | = d$$

Then, a leftist heap with a right chain of length d must conatin a complete bianry tree of a height d. Furthermore, this leftist heap must contains $2^{d+1}-1$ nodes including $2^{d}-1$ internal nodes.<br><br>

In other word, if we have n nodes in a leftist heap
$$d \leq \left \lfloor log_{2}(n+1) \right \rfloor - 1 = O(logn)$$
</p>

#### 10.6.4 Merge
<p align="justify">
Leftist heap breaks structure of normal heap because of leftist property, so we take use of binary tree to realize leftist heap instead of vector.<br><br>

Suppose we have 2 leftist heaps with root A and B (A > B), we take A's right sub-heap and recursively merge it and B. Then we put the merged heap as A's right sub-heap. If NPL(A's left sub-heap) < NPL(new sight sub-heap), we swap the two.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/leftist_heap_1.png"/></center>
</p>
{% highlight C++ %}
template <typename T>
static BinNodePosi(T) merge(BinNodePosi(T) a, BinNodePosi(T) b)
{
	if (!a) return b;
	if (!b) return a;
	if (a->data <  b->data) swap(a, b);
	a->rc = merge(a->rc, b);
	a->rc->parent = a;
	if (!a->lc || a->lc->npl < a->rc->npl) swap(a->lc, a->rc);
	a->npl = a->rc?a->rc->npl+1:1;
	return a;
}
{% endhighlight %}

#### 10.6.5 Insert
<p align="justify">
In fact, insert is a merge, becaue we can regard a new node as a leftist heap with only one element.
</p>
{% highlight C++ %}
void PQ_LeftHeap<T>::insert (T e)
{
	BinNodePosi(T) v = new BinNode<T>(e);
	_root = merge(_root, v);
	_root->parent = NULL;
	_size++;
}
{% endhighlight %}

#### 10.6.6 DelMax
<p align="justify">
Similarly, remove is also a merge, because after we remove root node, we only merge its two sub-heap.
</p>
{% highlight C++ %}
void PQ_LeftHeap<T>::delMax ()
{
	BinNodePosi(T) lHeap = _root->lc;
	BinNodePosi(T) rHeap = _root->rc;
	T e = _root->data;
	delete _root;
	_size--;
	_root = merge(lHeap, rHeap);
	if (_root) _root->parent = NULL;
	return e;
}
{% endhighlight %}


## 11. String
### 11.1 Definition
<p align="justify">
A string is a finit sequence with characters from alpabets. More generally, a string can contain an English article, a piece of C++ program and DNA sequence etc.
</p>

#### 11.1.1 Equality
<p align="justify">
If two strings S, T are equal, they must have a same length and S[i] = T[i] for i $\in [0, n)$.
</p>

#### 11.1.2 Sub-string
<p align="justify">
S.substr(i, k) = S[i, i+k), 0 $\leq i < n$, 0 $\leq$ k.<br><br>

Namely, k consecutive characters from s[i].
</p>

#### 11.1.3 Prefix
<p align="justify">
S.prefix(k) = S.substr(0, k) = S[0, k), o $\leq$ k $\leq$ n<br><br>

Namely, first k characters in S
</p>

#### 11.1.4 Suffix
<p align="justify">
S.suffix(k) = S.substr(n-k, k) = S[n-k, n), 0 $\leq$ k $\leq$ n<br><br>

Namely, last k characters in S
</p>

#### 11.1.5 Null String
<p align="justify">
Null string is a string with 0 length. 
</p>

### 11.2 Pattern Matching
<p align="justify">
Pattern string P and Text string T<br><br>

We need consider 4 questions about pattern matching:<br>
Detection, <b>location</b>, couting and enumeration.
</p>

#### 11.2.1 KMP
<p align="justify">
Knuth-Morris-Pratt<br><br>

If T[i] $\neq$ P[j], T[i-j, i] = P[0, j]. P can slide right t units instead of one.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/kmp_1.png"/></center>
</p>
{% highlight C++ %}
int KMP(char *P, char *T)
{
	int *next = buildNext(P);
	int n = (int)strlen(T), i = 0;
	int m = (int)strlen(P), j = 0;
	while(j < m && i < n)
	{
		if (0 > j || T[i] == P[j])
		{
			i++;
			j++;
		}
		else
		{
			j = next[j];
		}
	}
	delete []next;
	return i-j;
}
{% endhighlight %}
<p align="justify">
If i-j > n-m, there is no match between T and P, otherwise, a match starts at T[i-j].<br><br>

To understand <b>next</b> table<br><br>

In fact, we need to find t to replace y
$$N(P, j) = \{ 0 \leq t < j | P[0, t) == P[j-t, j) \}$$

It is possible to have more than 1 value satisfying our requirment. Among them, we pick the maximum t in order to avoid <b>backtracking</b>.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/kmp_2.png"/></center>
</p>
<p align="justify">
How to get <b>next</b> table?<br>
Suppose we have already next[j], we want to calculate next[j+1].
$$
\begin{matrix}
1 +\text{next}[j]\\
\\
1 + \text{next}[\text{next}[j]]\\
\\
1 + \text{next}[\text{next}[\text{next}[j]]]\\
\vdots
\end{matrix}
$$

We visit forward each item form j-1, until we find some item i, P[j+1] = P[i]. Besides, for the first character, next[0] = -1.
</p>
{% highlight C++ %}
int *buildNext(char *P)
{
	size_t m = strlen(P), j = 0;
	int *N = new int[m];
	int t = N[0] = -1;
	while (j < m - 1)
	{
		if (0 > t || P[j] == P[t])
		{
			N[++j] = ++t;
		}
		else
		{
			t = N[t];
		}
	}
	return N;
}
{% endhighlight %}
<p align="justify">
Amortized complexity is $O(n+m)$<br><br>

<b>Optimization</b> for next table<br>
If a pattern string is like this <b>00001</b> (ignore \0 in last position), its next table is such that [-1, 0, 1, 2, 3].<br><br>

We can observe next[3] = 2. Imagine a mismatch happens at $3^{th}$ position, there is definitely another mismatch at next[3] = $2^{th}$ position. In fact, next[3] should be -1. So, we can optimize our next table.
</p>
{% highlight C++ %}
int *buildNext(char *P)
{
	size_t m = strlen(P), j = 0;
	int *N = new int[m];
	int t = N[0] = -1;
	while (j < m - 1)
	{
		if (0 > t || P[j] == P[t])
		{
			j++;
			t++;
			N[j] = P[j]!=P[t]?t:N[t];
		}
		else
		{
			t = N[t];
		}
	}
	return N;
}
{% endhighlight %}

#### 11.2.2 BM
<p align="justify">
BM: Boyer–Moore<br><br>

<b>Bad-Character</b><br>
We compare T and P from the last one to the first one. A mismatch at some position is called bad character. Similar to KMP, BC['x'] depends on itself instead of T.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/bm_1.png"/></center>
</p>
<p align="justify">
If more than 1 successor of some a bad character, we pick up the one with the biggest rank in P (right-most). If no successor, we pick up [-1]. If one successor exists but in the right of x, this will make P move left. In this case, we move P right one cell.<br><br>

Construct bc table in $O(s)$ for space and $O(s+m)$ for time.
</p>
{% highlight C++ %}
int *buildBC(char *P)
{
	int *bc = new int[256]; // alphabet table
	for (size_t j = 0; j < 256; j++)
	{
		bc[j] = -1;
	}
	for (size_t m = strlen(P), j = 0; j < m; j++)
	{
		bc[P[j]] = j;
	}
	return bc;
}
{% endhighlight %}
<p align="justify">
Base on bc strategy:<br>
Best condition: $O(\frac{n}{m})$<br>
Worst condition: $O(n*m)$<br><br>

<b>Good-Suffix</b><br>
If we fail to match at 'X', we assume a suffix of 'X' must be matched.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/bm_2.png"/></center>
</p>
<p align="justify">
MS[j]: Among all suffix of P[0, j], the longest one mathcing with some a P's suffix.<br>
For example, P = "ICED RICE PRICE". P[8] = 'E', MS[8] = 'RICE' becasue 'RICE' matches with P'ssuffix 'RICE'.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/bm_3.png"/></center>
</p>
<p align="justify">
ss[j] = |MS[j]| = max{0 $\leq$ s $\leq$ j+1 | P(j-s, j] = P[m-s, m)} means a length of MS[j]. For eample, P[2] = 'E', MS[2] = 'ICE', ss[2] = 3.
</p>
<p align="justify">
From ss[] to gs[]<br>
If ss[j] = j+1, MS[j] must be a prefix of P. If we fail to match at i (i < m-j-1), [m-j-1, m) must be already matched, gs[i] = m-j-1, which means move P right (m-j-1).<br>
If ss[j] $\leq$ j, we fail to match at m-ss[j]-1, gs[m-ss[j]-1] = m-j-1.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/bm_4.png"/></center>
</p>
<p align="justify">
In space: bc[] + gs[] = $O(\Sigma + m)$, where $\Sigma$ is a alphabet size.<br><br>

In time:<br>
Best condition: $O(\frac{n}{m})$<br>
Worst condition: $O(n+m)$
</p>

#### 11.2.3 Karp-Rabin
<p align="justify">
God made the integers; all else is the work of man. -- L. Kronecker<br><br>

We can convert a string into an interger, then comparing two string becomes comparing two integers in $O(1)$.<br><br>

Suppose we have an alphabet $|\Sigma| = d$, we can regard a character in $\Sigma$ a d-unit number. For example, English alphabet (uppercase)
$$\Sigma = \{ A, B, C, \cdots, Z \}$$

CAT = 2 0 19<br><br>

If this alphabet is big enough, a string's fingerprint is extremely large so that we cannot compare two integers in $O(1)$. At this time, we take use if hash table.
</p>


## 12. Sort
<p align="justify">
<table class="c">
  <tr><th>Sort realm</th><th>Best complexity</th><th>Average complexity</th><th>Worst complexity</th><th>Extra space</th><th>Is stable</th></tr>
  <tr><td>Select Sort</td><td>$O(n^{2})$</td><td>$O(n^{2})$</td><td>$O(n^{2})$</td><td>$O(1)$</td><td>$X$</td></tr>
  <tr><td>Bubble Sort</td><td>$O(n)$</td><td>$O(n^{2})$</td><td>$O(n^{2})$</td><td>$O(1)$</td><td>$\checkmark$</td></tr>
  <tr><td>Merge Sort</td><td>$O(nlogn)$</td><td>$O(nlogn)$</td><td>$O(nlogn)$</td><td>$O(n)$</td><td>$\checkmark$</td></tr>
  <tr><td>Insert Sort</td><td>$O(n)$</td><td>$O(n^{2})$</td><td>$O(n^{2})$</td><td>$O(1)$</td><td>$\checkmark$</td></tr>
  <tr><td>Bucket Sort</td><td>$O(n+m)$</td><td>$O(n+m)$</td><td>$O(n+m)$</td><td>$O(n+m)$</td><td>$\checkmark$</td></tr>
  <tr><td>Heap Sort</td><td>$O(nlogn)$</td><td>$O(nlogn)$</td><td>$O(nlogn)$</td><td>$O(n)$</td><td>$X$</td></tr>
  <tr><td>Quick Sort</td><td>$O(nlogn)$</td><td>$O(nlogn)$</td><td>$O(n^{2})$</td><td>$O(1)$</td><td>$X$</td></tr>
  <tr><td>Shell Sort</td><td>$n^{1.5}$</td><td>$n^{1.5}$</td><td>$n^{1.5}$</td><td>$O(1)$</td><td>$X$</td></tr>
</table>
</p>

### 12.1 Select Sort
<p align="justify">
Two steps: select + put. That is to say, <br>
(1) at each time, we select a biggest one (or a samllest one), then we put the selected on in our container.<br>
(2) repeat (1)<br><br>

<b>Complexity</b><br>
Both the best condition and the worst condition: $\Theta(n^{2})$<br>
The most time is spent on the first of select, because at each time we have to vist all elements once to find a maximun, then we repeat this step.
$$O(n + (n-1)+...+1)=O(n^{2})$$
</p>

### 12.2 Bubble Sort
<p align="justify">
(1) Scan all element if two near element form an inversed pair, swap them.<br>
(2) Repeat (1) until all elements are in order.
</p>
{% highlight C++ %}
void BubbleSort(int A[], int n)
{
	for (bool started = false; sorted = !sorted; n--)
	{
		for (int i = 1; i < n; i++)
		{
			if (A[i-1] > A[i])
			{
				swap(A[i-1], A[i]);
				sorted = false;
			}
		}
	}
}
{% endhighlight %}

<p align="justify">
If our array is sorted at some iteration, we can break it.
</p>
{% highlight C++ %}
template <typename T>
void vector<T>::bubbleSort(Rank lo, Rank hi)
{
	while (!bubble(lo, hi--));
}

template <typename T>
bool vector<T>::bubble(Rank lo, Rank hi)
{
	bool sorted = true;
	while (++lo < hi)
	{
		if (_elem[lo-1] > _elem[lo])
		{
			sorted = false;
			swap(_elem[lo-1], _elem[lo]);
		}
	}
	return sorted;
}
{% endhighlight %}

### 12.3 Merge Sort
<p align="justify">
(1) Divide an array into 2 sub-array recursively until each sub-array has 1 element<br>
(2) Merge 2 ordered arrays
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/mege_sort_1.png"/></center>
</p>
<p align="justify">
$$T(n) = 2T(\frac{n}{2}) + O(n)$$

Complexity is in $O(nlogn)$.
</p>
{% highlight C++ %}
template <typename T>
void vector<T>::mergeSort(Rank lo, Rank hi)
{
	if (hi - lo < 2) return;
	int mi = (lo + hi) >> 1;
	mergeSort(lo, mi);
	mergeSort(mi, hi);
	merge(lo, mi, hi);
}
template <typename T>
void vector<T>::merge(Rank lo, Rank mi, Rank hi)
{
	T *A = _elem + lo;
	int lb = mi - lo;
	T *B = new T[b];
	for (Rank i = 0; i < lb; B[i]=A[i++]);
	int lc = hi - mi;
	T *C = _elem + mi;
	for (Rank i = 0, j = 0, k = 0; (j < lb) || (k < lc); )
	{
		if ((j < lb) && (lc <= k || (B[i] <= C[k]))) A[i++] = B[j++];
		if ((k < lc) && (lb <= j || (C[k] < B[j]))) A[i++] = C[k++];
	}
	delete []B;
}
{% endhighlight %}

### 12.4 Insert Sort
<p align="justify">
We have two containers, one unsorted container an one sorted container. Initially, the sorted container is null.<br>
(1) Pick up one element from the unsorted container (usually we choose the first one)<br>
(2) Insert the selected element into the sorted container by finding its proper position (its value is between its last one and next one)<br>
(3) Repeat (1) and (2)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/insert_sort1.png"/></center>
</p>
<p align="justify">
Instead of using two container, we can only use one container to implement this algorithm. Precisely, when we enter into the $r^{th}$ position, that is to say, segment [0, r) is well sorted. At this time, we pick up $r^{th}$ element a, we put it into the segment [0, r]. Suppose we find a proper position k to insert a, we have $k \in [0, r]$. So, we put a in the position k and push all elements in [k, r) backward one cell. To implement this algorithm, it is better to take list structure.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/insert_sort2.png"/></center>
</p>
<p align="justify">
<b>Complexity</b><br>
The best condition: $O(n)$. Because [0, r) is sorted, we only need append a, this take O(n)<br>
The worst condition: $O(n^{2})$. At each time, $r^{th}$ element a will compare each element in [0, r). Here, we use List structure instead of vector, so we must compare one by one.<br><br>

<b>Mean complexity</b><br>
hypothesis: All elements are Independent and identically distributed.<br>
For some iteration r, the time to insert [r] into [0, r] is<br>
$$\frac{r+(r-1)+...+1+0}{r+1}+1=\frac{r}{2}+1$$

So, total expectation is
$$E[\frac{r+(r-1)+...+1+0}{r+1}+1] = E[\frac{r}{2}+1] = E[\frac{0+1+...+n-1}{2}] + 1 = O(n^{2})$$

<b>Input sensitive</b><br>
For one sequence, maximum number of reversed pairs is<br>
$$C_{n}^{2} = \frac{n\cdot (n-1)}{2}$$

For some iteration r, [r] = a. Before r, suppose the number of revsered pari is I, then <br>
$$O(I+n)$$

which denotes I compares and n inserts
</p>

### 12.5 Bucket Sort
<p align="justify">
We apply hash table to sorting. If we know a range of unsorted number, for example, 26 english alphabets. We can etablish a bucket array called count with a length of 26. Besides, we prepare another array called accum with a same size. In advance, we define A to Z by 0 to 25, which corresponds to our bucket array.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/bucket_sort_1.png"/></center>
</p>
<p align="justify">
We scan an unsorted array, we put each alphabet into our bucket array by counting its appearance time. For example, 'A' for count[0]++. Then we traverse from 0 to 25 to update accum: accum[i] = accum[i-1] + count[i] for i = 1 to 25 and accum[0] = count[0].<br><br>

In order to get a sorted array, we traverse from 0 to 25 again. For example, count['B'] = 1, accum['B'] = 1, represent an interval [0, 1) should be B; count['J'] = 2, accum['J'] = 14, show [12, 14) should be J.<br><br>

Complexity is in $O(n+m)$
</p>

### 12.6 Heap Sort
<p align="justify">
(1) Heapification for n elements in $O(n)$<br>
(2) Pick root node and delete it in O(logn)<br>
(3) Repeat (2)<br><br>

Time is in $O(n+log(n!)) = O(nlogn)$
$$log(n!) = log(1) + log(2) + \cdots + log(n) \leq log(n) + log(n) + \cdots + log(n) = nlog(n)$$
$$log(n!) \geq log(\frac{n}{2}) + log(\frac{n}{2}+1) + \cdots + log(n) \geq log(\frac{n}{2}) + log(\frac{n}{2}) + \cdots + log(\frac{n}{2}) = \frac{n}{2}log(\frac{n}{2}) $$
</p>

### 12.7 Quick Sort
<p align="justify">
Divide a sequence S into two sub-sequence
$$S = S_{1} + S_{2}$$

$$max(S_{1}) \leq min(S_{2})$$

If two sub-sequence is sorted, S will be sorted.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/quick_sort_1.png"/></center>
</p>
{% highlight C++ %}
template <typename T>
void vector<T>::quickOrder(Rank lo, Rank hi)
{
	if (hi - lo < 2) return; // base case
	Rank mi = partition(lo, hi - 1);
	quickSort(lo, mi);
	quickSort(mi+1, hi);
}
{% endhighlight %}
<p align="justify">
Construct a pivot:<br>
(1) Select the first one as candidate for pivot<br>
(2) Prepare two indicator lo, hi pontting to first element and last element<br>
(3) When lo is povot,  if A[hi] < A[lo], swap(A[hi], A[lo]) and set hi as povot, otherwise, hi--; when hi is pivot, if A[lo] > A[hi], swap(A[lo], A[hi]) and set lo as pivot, otherwise, lo++
</p>
{% highlight C++ %}
void quickSort(int *pnt, int start, int end)
{
    int i, j;
    i = start;
    j = end;
    if (start < end)
    {
        while(i < j)
        {
            while(pnt[i] <= pnt[j] && i < j)
            {
                j--;
            }
            swap(pnt, i, j);
            while(pnt[i]<=pnt[j] && i < j)
            {
                i++;
            }
            swap(pnt, i, j);
        }
        quickSort(pnt, start, i-1);
        quickSort(pnt, i+1, end);
    }
}
{% endhighlight %}
<p align="justify">
QuickSort is unstable because it is possible to inverse a left number and a right number. Space is in $O(a)$. Best condition in $O(nlogn)$, worst condition in $O(n^{2})$<br><br>

But average complexity is in $O(nlogn)$
$$T(n) = (n + 1) + \frac{1}{n} = \sum_{k=0}^{n-1}[T(k) + T(n-k-1)] = (n + 1) + \frac{2}{n} \sum_{k=0}^{n-1}T(k)$$

$$nT(n) = n(n+1) + 2\sum_{k=0}^{n-1}T(k)$$

$$(n-1)T(n-1) = (n-1)n + 2\sum_{k=0}^{n-2}T(k)$$

$$nT(n) - (n-1)T(n-1) = 2n + 2T(n-1)$$

$$nT(n) - (n+1)T(n-1) = 2n$$

$$\frac{T(n)}{n+1} = \frac{2}{n+1} + \frac{T(n-1)}{n} = \frac{2}{n+1} + \frac{2}{n} + \frac{T(n-2)}{n-1}$$

$$= \frac{2}{n+1} + \frac{2}{n} + \frac{2}{n-1} + \cdots + \frac{2}{2} + \frac{T(0)}{1}$$

$$= (2ln2)logn = 1.39logn$$

Another version:<br>
We divide a sequence S into 4 parts
$$S = [ lo ] + L(lo, mi] + G(mi, k) + U[k, hi]$$

Similarly, we want to find a pivot
$$L < pivot \leq G$$

If Spivot \leq $[ k ]$, put k into G (k++); otherwise, swap(S[ ++mi ], S[ k++ ]).
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/quick_sort_2.png"/></center>
</p>
{% highlight C++ %}
template <typename T>
Rank vector<T>::partition(Rank lo, Rank hi)
{
	swap(_elem[lo], _elem[lo+rand()%(hi-lo+1)]);
	T pivot = _elem[lo];
	int mi = lo;
	for (int k = lo+1; k <= hi; k++)
	{
		if (_elem[k] < pivot)
		{
			swap(_elem[++mi], _elem[k]);
		}
	}
	swap(_elem[lo], _elem[mi]);
	return mi;
}
{% endhighlight %}
<p align="justify">
The quicksort can adopt different strategies when selecting pivot. This question attempts to use an example to illustrate that the strategy of “choose the middle one of three elements” tends to obtain a more balanced pivot than the randomly selected strategy<br><br>

Let the length of the sequence to be sorted n be large, if the selection of the pivot makes the length ratio of the long/short subsequences after the partition greater than 9:1, it is called unbalanced<br><br>

For different pivot selection strategies, estimate the probability of imbalance<br>
Select one randomly from the n elements as the pivot: 0.2<br>
Select three elements at same probability from n elements with their intermediate elements as pivot:0.056<br><br>

Consider a cube (0, 0, 0) $\rightarrow$ (1, 1, 1). Each axe has three parts a (0, 0.1), b (0, 1, 0.9), c (0.9, 1).<br>
Unbalanced space is 1 - (0.8 * 0.1 * 0.1 * 6+ 0.8 * 0.8 * 0.1 * 6+0.8^3) = 1 - 0.944 = 0.056
</p>
<p align="justify">
<b>Select mode</b><br>
Consider vector A has a prefix P (|P| is even), P has an element x which exactly appears $\frac{|P|}{2}$ in P. If A-P has a mod m, A has a mod m.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/select_mode_1.png"/></center>
</p>
{% highlight C++ %}
template <typename T> T majEleCandidate (vector<T> A)
{
	T maj;
	for (int c = 0, i = 0; i < A.size(); i++)
	{
		maj = A[i];
		c = 1;
	}
	else
	{
		maj == A[i]?c++:c--;
	}
	return maj;
}
{% endhighlight %}
<p align="justify">
<b>Quick select</b><br>
Consider an unsorted sequence, we can find its povot x, if a target element < x, turn to L; otherwise, turn to G.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/quick_select_1.png"/></center>
</p>
{% highlight C++ %}
template <typename T>
void quickSelect(vector<T> & A, Rank k)
{
	for (Rank lo = 0, hi = A.szie()-1; lo < hi; )
	{
		Rank i = lo, j = hi;
		T pivot = A[lo];
		while(i < j)
		{
			while(i < j && pivot <= A[j]) j--; A[i] = A[j];
			while(i < j && A[i] <= pivot) i++; A[j] = A[i];
		}
		A[i] = pivot;
		if (k <= i) hi = i - 1;
		if (i <= k) lo = i + 1;
	}
}
{% endhighlight %}
<p align="justify">
<b>Linear select</b><br>
Q is a small constant<br>
(1) if n = |A| < Q return quickSelect(A, k) or others; otherwise, divide A evenly into $\frac{n}{Q}$ sub-sequences with a size of Q<br>
(2) sort each sub-sequence and determine $\frac{n}{Q}$ median<br>
(3) among this medians, call linearSort() to find a median M by recursion<br>
(4) classify all elements according to M: L / E / G = { x < / = / > M | x $\in$ A}<br>
(5)<br>
if k $\leq$ |L|, return linearSelect(L, k)<br>
if k $\leq$ |L| + |E|, return M<br>
return linearSelect(G, k-|L|-|E|)
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/DSA/linear_select_1.png"/></center>
</p>
<p align="justify">
Complexity:<br>
$$T(n) = O(n) + T(\frac{n}{Q}) + T(\frac{3n}{4})$$

In order to guarantee linear function
$$\frac{n}{Q} + \frac{3n}{4} < n$$

$$\frac{1}{Q} + \frac{3}{4} < 1$$

Make Q = 5
$$T(n) = cn + T(\frac{n}{5}) + T(\frac{3n}{4}) = O(n)$$
</p>

### 12.8 Shell Sort
<p align="justify">
Regard a sequence as a matrix, sort each column. Shell sort is also called w-sort.<br> If column w is sorted, we call w-sorted. Diminish w and repeat until w = 0.<br><br>

Insert sort is good for sorting each column.<br><br>

h-ordered: let h $\in$ N, a sequence S[0, n)is h-ordered if S[ i ] $\leq$ S[ i+h ] for 0 $\leq$ i < n-h.<br><br>

A 1-ordered sequence is sorted.<br><br>

h-sorting: an h-ordered sequence is obtained by<br>
(1) arranging S into a 2D matrix with h columns<br>
(2) sorting each column respectively<br><br>

Theorem k -- Knuth<br>
A g-ordered sequence remains g-ordered after being h-sorted.<br><br>

PS, Pratt, Sedgewick
</p>


## 13. Programming Exercises
### 13.1 Range
<p align="justify">
<b>Descriptioin</b><br>
Let S be a set of n integral points on the x-axis. For each given interval [a, b], you are asked to count the points lying inside.<br><br>

<b>Input</b><br>
The first line contains two integers: n (size of S) and m (the number of queries). The second line enumerates all the n points in S. Each of the following m lines consists of two integers a and b and defines an query interval [a, b].<br><br>

<b>Output</b><br>
The number of points in S lying inside each of the m query intervals.<br><br>

<b>Example</b><br>
Input<br>
5 2<br>
1 3 7 9 11<br>
Output<br>
o<br>
3<br><br>

<b>Example</b><br>
0 <= n, m <= 5 * 10^5<br>
For each query interval [a, b], it is guaranteed that a <= b.<br>
Points in S are distinct from each other.<br>
Coordinates of each point as well as the query interval boundaries a and b are non-negative integers not greater than 10^7.<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Solution</b><br>
1) Sort all points with QuickSort<br>
2) Locate the left border's index ceil[a] and the right border's index floor[b].<br>
The final answer is floor[b]-ceil[a]+1.
</p>

### 13.2 Zuma
<p align="justify">
<b>Description</b><br>
Let's play the game Zuma! There are a sequence of beads on a track at the right beginning. All the beads are colored but no three adjacent ones are allowed to be with a same color. You can then insert beads one by one into the sequence. Once three (or more) beads with a same color become adjacent due to an insertion, they will vanish immediately.<br>
Note that it is possible for such a case to happen for more than once for a single insertion. You can't insert the next bead until all the eliminations have been done.<br>
Given both the initial sequence and the insertion series, you are now asked by the fans to provide a playback tool for replaying their games. In other words, the sequence of beads after all possible eliminations as a result of each insertion should be calculated.<br><br>

<b>Input</b><br>
The first line gives the initial bead sequence. Namely, it is a string of capital letters from 'A' to 'Z', where different letters correspond to beads with different colors.<br>
The second line just consists of a single interger n, i.e., the number of insertions.<br>
The following n lines tell all the insertions in turn. Each contains an integer k and a capital letter Σ, giving the rank and the color of the next bead to be inserted respectively. Specifically, k ranges from 0 to m when there are currently m beads on the track.<br><br>

<b>Output</b><br>
n lines of capital letters, i.e., the evolutionary history of the bead sequence.<br>
Specially, "-" stands for an empty sequence.<br><br>

<b>Example</b><br>
Input<br>
ACCBA<br>
5<br>
1 B<br>
0 A<br>
2 B<br>
4 C<br>
0 A<br>
Output<br>
ABCCBA<br>
AABCCBA<br>
AABBCCBA<br>
-<br>
A<br><br>

<b>Restrictions</b><br>
0 <= n <= 10^4<br>
0 <= length of the initial sequence <= 10^4<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Solution</b><br>
Attetion: both 3 beads' elimination and 4 beads' elimination are possible
</p>

### 13.3 LightHouse
<p align="justify">
<b>Description</b><br>
As shown in the following figure, If another lighthouse is in gray area, they can beacon each other. For example, in following figure, (B, R) is a pair of lighthouse which can beacon each other, while (B, G), (R, G) are NOT.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ALGOS/DSA/13_3_1.png"/></center>
</p>
<p align="justify">
<b>Input</b><br>
1st line: N<br>
2nd ~ (N + 1)th line: each line is X Y, means a lighthouse is on the point (X, Y).<br><br>

<b>Output</b><br>
How many pairs of lighthourses can beacon each other ( For every lighthouses, X coordinates won't be the same , Y coordinates won't be the same )<br><br>

<b>Example</b><br>
Input<br>
3<br>
2 2<br>
4 3<br>
5 1<br>
Output<br>
1<br><br>

<b>Restrictions</b><br>
For 90% test cases: 1 <= n <= 3 * 10^5<br>
For 95% test cases: 1 <= n <= 10^6<br>
For all test cases: 1 <= n <= 4 * 10^6<br>
For every lighthouses, X coordinates won't be the same , Y coordinates won't be the same.<br>
1 <= x, y <= 10^8<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Hints</b><br>
The range of int is usually [-2^31, 2^31 - 1], it may be too small.<br><br>

<b>Solution</b><br>
This problem can be converted into an inverted sort pair problem. How many pairs of LightHouse can beacon each other means how many non-inversable pairs.<br>
1) Sort all points(x, y) according to the coordinate x.<br>
2) Divide-and-conquer. MergeSort all points according to the coordinate y, at the same time keep record of the number of inversable pairs I. $C_{n}^{2}-I$ is the final answer.
</p>

### 13.4 Train
<p align="justify">
<b>Description</b><br>
Figure shows the structure of a station for train dispatching.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ALGOS/DSA/13_4_1.png"/></center>
</p>
<p align="justify">
In this station, A is the entrance for each train and B is the exit. S is the transfer end. All single tracks are one-way, which means that the train can enter the station from A to S, and pull out from S to B. Note that the overtaking is not allowed. Because the compartments can reside in S, the order that they pull out at B may differ from that they enter at A. However, because of the limited capacity of S, no more that m compartments can reside at S simultaneously.<br>
Assume that a train consist of n compartments labeled {1, 2, …, n}. A dispatcher wants to know whether these compartments can pull out at B in the order of {a1, a2, …, an} (a sequence). If can, in what order he should operate it?<br><br>

<b>Input</b><br>
Two lines:<br>
1st line: two integers n and m;<br>
2nd line: n integers separated by spaces, which is a permutation of {1, 2, …, n}. This is a compartment sequence that is to be judged regarding the feasibility.<br><br>

<b>Output</b><br>
How many pairs of lighthourses can beacon each other ( For every lighthouses, X coordinates won't be the same , Y coordinates won't be the same )<br><br>

<b>Example</b><br>
Input<br>
5 2<br>
1 2 3 5 4<br>
Output<br>
push<br>
pop<br>
push<br>
pop<br>
push<br>
pop<br>
push<br>
push<br>
pop<br>
pop<br><br>

Input<br>
5 5<br>
3 1 2 4 5<br>
Output<br>
No<br><br>

<b>Restrictions</b><br>
1 <= n <= 1,600,000<br>
0 <= m <= 1,600,000<br>
Time: 2 sec<br>
Memory: 256 MB You can only use the C++ language. STL is forbidden.<br><br>

<b>Solution</b><br>
This is a question of stack permutation. We repeat this process, if some element is not corresponded, output no; otherwise, print all process.
</p>

### 13.5 Proper Rebuild
<p align="justify">
<b>Description</b><br>
In general, given the preorder traversal sequence and postorder traversal sequence of a binary tree, we cannot determine the binary tree.
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ALGOS/DSA/13_5_1.png"/></center>
</p>
<p align="justify">
For example, although they are two different binary tree, their preorder traversal sequence and postorder traversal sequence are both of the same.
But for one proper binary tree, in which each internal node has two sons, we can uniquely determine it through its given preorder traversal sequence and postorder traversal sequence.<br>
Label n nodes in one binary tree using the integers in [1, n], we would like to output the inorder traversal sequence of a binary tree through its preorder and postorder traversal sequence.<br><br>

<b>Input</b><br>
The 1st line is an integer n, i.e., the number of nodes in one given binary tree,<br>
The 2nd and 3rd lines are the given preorder and postorder traversal sequence respectively.<br><br>

<b>Output</b><br>
The inorder traversal sequence of the given binary tree in one line.<br><br>

<b>Example</b><br>
Input<br>
5<br>
1 2 4 5 3<br>
4 5 2 3 1<br>
Output<br>
4 2 5 1 3<br><br>

<b>Restrictions</b><br>
For 95% of the estimation, 1 <= n <= 1,000,00<br>
For 100% of the estimation, 1 <= n <= 4,000,000<br>
The input sequence is a permutation of {1,2...n}, corresponding to a legal binary tree.<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Hints</b><br>
<center><img src="https://raw.githubusercontent.com/chaopan95/chaopan95.github.io/master/_imgs/ALGOS/DSA/13_5_2.png"/></center>
</p>
<p align="justify">
observe the positions of the left and right children in preorder and postorder traversal sequence.<br><br>

<b>Solution</b><br>
Find L and R child for each node, in-order print
</p>

### 13.6 TSP
<p align="justify">
<b>Descriptioin</b><br>
Shrek is a postman working in the mountain, whose routine work is sending mail to n villages. Unfortunately, road between villages is out of repair for long time, such that some road is one-way road. There are even some villages that can’t be reached from any other village. In such a case, we only hope as many villages can receive mails as possible.<br>
Shrek hopes to choose a village A as starting point (He will be air-dropped to this location), then pass by as many villages as possible. Finally, Shrek will arrived at village B. In the travelling process, each villages is only passed by once. You should help Shrek to design the travel route.<br><br>

<b>Input</b><br>
There are 2 integers, n and m, in first line. Stand for number of village and number of road respectively.<br>
In the following m line, m road is given by identity of villages on two terminals. From v1 to v2. The identity of village is in range [1, n].<br><br>

<b>Output</b><br>
Output maximum number of villages Shrek can pass by.<br><br>

<b>Example</b><br>
Input<br>
4 3<br>
1 4<br>
2 4<br>
4 3<br>
Output<br>
3<br><br>

<b>Restrictions</b><br>
1 <= n <= 1,000,000<br>
0 <= m <= 1,000,000<br>
These is no loop road in the input.<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Hints</b><br>
Topological sorting<br><br>

<b>Solution</b><br>
Topological sort + dynamic programming
</p>

### 13.7 Broadcast
<p align="justify">
<b>Descriptioin</b><br>
A broadcaster wants to set up a radio broadcast transmitter in an area. There are n towns in the area, and each town has a transmitter installed and plays its own program.<br>
However, the company only licensed the two bands FM104.2 and FM98.6, and transmitters using the same band would interfere with each other. It is known that the signal coverage of each transmitter is a circular area with a radius of 20km. Therefore, if two towns with a distance of less than 20km use the same band, they will not be able to function properly due to band interference. Listen to the show. Now give a list of towns with distances less than 20km, and try to determine whether the company can make residents of the entire region hear the broadcasts normally.<br><br>

<b>Input</b><br>
The first line is two integers n, m, which are the number of towns and the number of town pairs that are less than 20km. The next m lines, 2 integers per line, indicate that the distance between the two towns is less than 20km (numbering starts at 1).<br><br>

<b>Output</b><br>
Output 1 if the requirement is met, otherwise -1.<br><br>

<b>Example</b><br>
Input<br>
4 3<br>
1 2<br>
1 3<br>
2 4<br>
Output<br>
1<br><br>

<b>Restrictions</b><br>
1 ≤ n ≤ 10000<br>
1 ≤ m ≤ 30000<br>
There is no need to consider the spatial characteristics of a given 20km town list, such as whether triangle inequality is satisfied, whether more information can be derived using transitivity, and so on.<br>
Time: 2 sec<br>
Space: 256MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Tips</b><br>
BFS<br><br>

<b>Solution</b><br>
Once three towns have a distance less than 20km to each other, return -1.
</p>

### 13.12 Cycle
<p align="justify">
<b>Descriptioin</b><br>
Cycle shifting refers to following operation on the sting. Moving first letter to the end and keeping rest part of the string. For example, apply cycle shifting on ABCD will generate BCDA. Given any two strings, to judge if arbitrary times of cycle shifting on one string can generate the other one.<br><br>

<b>Input</b><br>
There m lines in the input, while each one consists of two strings separated by space. Each string only contains uppercase letter 'A'~'Z'.<br><br>

<b>Output</b><br>
For each line in input, output YES in case one string can be transformed into the other by cycle shifting, otherwise output NO.<br><br>

<b>Example</b><br>
Input<br>
AACD CDAA<br>
ABCDEFG EFGABCD<br>
ABCD ACBD<br>
ABCDEFEG ABCDEE<br>
Output<br>
YES<br>
YES<br>
NO<br>
NO<br><br>

<b>Restrictions</b><br>
0 <= m <= 5000<br>
1 <= |S1|, |S2| <= 10^5<br>
Time: 2 sec<br>
Memory: 256 MB<br>
You can only use the C++ language. STL is forbidden.<br><br>

<b>Solution</b><br>
According to its definition strcopy s1's prefix to s1's suffix.
</p>
