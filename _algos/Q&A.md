---
layout: post
title:  "Q & A"
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


## 1. C++
<p align="justify">
<b>How to new a 2D matrix?</b><br>
</p>
{% highlight C++ %}
int n = 5;
int **g = new int *[n];
for (int i = 0; i < n; i++)
{
    g[i] = new int[n];
}
for (int i = 0; i < n; i++)
{
    delete []g[i];
}
delete []g;
{% endhighlight %}

<p align="justify">
<b>How to convert a char to string like 'a' to "a"?</b><br>
</p>
{% highlight C++ %}
string res = "";
res.push_back('a');  // res = "a"
res += 'b';  // res = "ab"
res.append(1, 'c');  // res = "abc"
res.assign(1, 'd');  // res = "d";
res.insert(1, 1, 'e');  // res = "de"
res.insert(0, 1, 'e');  // res = "ed"
res.replace(0, 1, 1, 'f');  // res = "fe"
string res(2, 'a');  // res = "aa"
{% endhighlight %}

<p align="justify">
<b>How to print an inversed string in a recursive way?</b><br>
</p>
{% highlight C++ %}
void printInverseStrRecur(string str, int idx, int len)
{
    if (idx < len)
    {
        printInverseStrRecur(str, idx+1, len);
        cout << str[idx];
    }
}
{% endhighlight %}

<p align="justify">
<b>How to generate a combination of k from n elements?</b><a href="https://stackoverflow.com/questions/9430568/generating-combinations-in-c"> [1]</a>
</p>
{% highlight C++ %}
void combine(int *arr, int n, int k, int idx,
             bool *isVisited, int count)
{
    if (count == k)
    {
        for (int i = 0; i < n; i++)
        {
            if (isVisited[i]) { printf("%d", i); }
        }
        printf("\n");
    }
    else if (idx < n)
    {
        isVisited[idx] = true;
        combine(arr, n, k, idx+1, isVisited, count+1);
        isVisited[idx] = false;
        combine(arr, n, k, idx+1, isVisited, count);
    }
}

void combination(int *arr, int n, int k)
{
    bool *isVisited = new bool [n]{};
    combine(arr, n, k, 0, isVisited, 0);
    delete []isVisited;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int n = 6, k = 3, *arr = new int [n]{};
    for (int i = 0; i < n; i++) { arr[i] = i + 1; }
    combination(arr, n, k);
    delete []arr;
    return 0;
}
{% endhighlight %}

<p align="justify">
二项分布，n次试验，每次试验成功的概率为p，求问连续k次成功的概率为多少？<br>
例如：<br>
input: n = 2, k = 1, p = 0.5<br>
output: 0.75<br>
input: n = 10, k = 3, p = 0.4<br>
output: 0.3141
</p>
{% highlight C++ %}
void calProb(bool *dp, int n, int k, float p, int idx, float &res)
{
    if (idx == n)
    {
        for (int i = 0; i < n; i++) { printf("%d\t", dp[i]); }
        printf("\n");
        bool isOk = false;
        for (int i = 0; i < n-k+1; i++)
        {
            bool isK = true;
            for (int j = i; j < i+k; j++)
            {
                if (!dp[j])
                {
                    isK = false;
                    break;
                }
            }
            if (isK)
            {
                isOk = true;
                break;
            }
        }
        if (isOk)
        {
            float prob = 1.0;
            for (int i = 0; i < n; i++)
            {
                if (dp[i]) { prob *= p; }
                else { prob *= (1-p); }
            }
            printf("%f\n", prob);
            res += prob;
        }
    }
    else
    {
        dp[idx] = true;
        path(dp, n, k, p, idx+1, res);
        dp[idx] = false;
        path(dp, n, k, p, idx+1, res);
    }
}

int amin(int argc, const char * argv[])
{
    int n = 10, k = 3;
    float p = 0.4, res = 0.0;
    scanf("%d%d%f", &n, &k, &p);
    bool *dp = new bool [n]{};
    calProb(dp, n, k, p, 0, res);
    delete []dp;
    printf("%.4f\n", res);
    return 0;
}
{% endhighlight %}

<p align="justify">
We have 4 numbers. Apply four basic operator (+, -, *, /) to get 24.
</p>
{% highlight C++ %}
#include<iostream>
#include<string>
#include<vector>
#include<stack>
#include<queue>
#include<cmath>
#include<iomanip>
using namespace std;


class Solution {
public:
    /**
     *
     * @param arr int整型一维数组
     * @param arrLen int arr数组长度
     * @return bool布尔型
     */
    bool Game24Points(int* arr, int arrLen) {
        // write code here
        int n = arrLen;
        if (n != 4) { return false; }
        vector<vector<int>> all;
        permute(arr, all, n, 0);
        /*
        for (int i = 0; i < int(all.size()); i++)
        {
            for (int j = 0; j < int(all[i].size()); j++)
            {
                cout << all[i][j] << '\t';
            }
            cout << endl;
        }
        */
        vector<string> allOpt = allOperators();
        for (int i = 0; i < int(all.size()); i++)
        {
            int *a = new int [4]{};
            a[0] = all[i][0];
            a[1] = all[i][1];
            a[2] = all[i][2];
            a[3] = all[i][3];
            for (int j = 0; j < int(allOpt.size()); j++)
            {
                if (isOK(a, allOpt[j], n))
                {
                    return true;
                }
            }
            delete []a;
        }
        return false;
    }
    vector<string> allOperators()
    {
        vector<string> res;
        string opt = "+-*/";
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    string temp = "";
                    temp.push_back(opt[i]);
                    temp.push_back(opt[j]);
                    temp.push_back(opt[k]);
                    res.push_back(temp);
                }
            }
        }
        return res;
    }
    bool isOK(int *arr, string opt, int n)
    {
        stack<int> ns;
        stack<char> os;
        ns.push(arr[0]);
        int temp = 0;
        for (int i = 0; i < 3; i++)
        {
            if (opt[i] == '*')
            {
                temp = ns.top();
                ns.pop();
                ns.push(temp * arr[i+1]);
            }
            else if (opt[i] == '/')
            {
                if (ns.top() % arr[i+1] ) { return false; }
                temp = ns.top();
                ns.pop();
                ns.push(temp / arr[i+1]);
            }
            else
            {
                ns.push(arr[i+1]);
                os.push(opt[i]);
            }
        }
        stack<int> nns;
        stack<char> nos;
        while (!ns.empty())
        {
            nns.push(ns.top());
            ns.pop();
        }
        while (!os.empty())
        {
            nos.push(os.top());
            os.pop();
        }
        while (!nos.empty())
        {
            int a = nns.top();
            nns.pop();
            int b = nns.top();
            nns.pop();
            if (nos.top() == '+') { temp = a + b; }
            else if (nos.top() == '-') { temp = a - b; }
            else if (nos.top() == '*') { temp = a * b; }
            else
            {
                if (a % b) { return false; }
                temp = a / b;
            }
            nos.pop();
            nns.push(temp);
        }
        cout << nns.top() << endl;
        return nns.top() == 24;
    }
    void permute(int *arr, vector<vector<int>> &all, int n, int idx)
    {
        if (idx == n-1)
        {
            vector<int> temp;
            for (int i = 0; i < n; i++)
            {
                temp.push_back(arr[i]);
            }
            all.push_back(temp);
        }
        else
        {
            for (int i = idx; i < n; i++)
            {
                swap(arr, idx, i);
                permute(arr, all, n, idx+1);
                swap(arr, idx, i);
            }
        }
    }
    void swap(int *arr, int i, int j)
    {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
};


int main(int argc, const char * argv[]) {
    // insert code here...
    Solution sol;
    int *arr = new int [4]{};
    arr[0] = 7;
    arr[1] = 2;
    arr[2] = 1;
    arr[3] = 10;
    //cout << sol.isOK(arr, "*++", 4) << endl;
    //cout << sol.Game24Points(arr, 4) << endl;
    cout << sol.isOK(arr, "---", 4) << endl;
    return 0;
}
{% endhighlight %}


## 2. Java
<p align="justify">
<b>How to call a function?</b><br>
$\bigstar$ How to call a function in the same package?<br>
Consider we have a package called <b>P</b>, which have two files <b>file1.java</b> and <b>file2.java</b>.<br><br>

In file1, we want to import file2
</p>
{% highlight Java %}
Package P
import P.file2
public class file1 {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}
}
{% endhighlight %}

<p align="justify">
$\bigstar$ How to call a function in another package?<br>
If I have two packages called <b>P1</b> and <b>P2</b>, P1 has a file <b>file1.java</b> and P2 has a file <b>file2.java</b><br><br>

In file1, we want to import file2
</p>
{% highlight Java %}
Package P1
import P2.file2
public class file1 {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}
}
{% endhighlight %}

## 3. Python
<p align="justify">
<b>How to install un package in Anaconda by setup.py</b><br>
conda list<br>
/anaconda/lib/site-packages<br>
cd enter the directory site-packages<br>
python setup.py install<br><br>

<b>In fun(*args,**kwargs), what do *args and **kwargs mean?</b><br>
*args and **kwargs can be used as a container for infinitive arguments, namely, an uncertain number of arguments. In details, *args is for values without keys or variable names; while **kwargs is for key-value arguments.
</p>
{% highlight Python %}
def fun(*args):
    for x in args:
        print(x)

fun(1, 'abc', [1, 2, 3])
#Output
#1
#abc
#[1, 2, 3]

def fun(**kwargs):
    for k, v in kwargs.items():
        print(k, v)

fun(a1=1, a2='abc', a3=[1, 2, 3])
#Output
#a1 1
#a2 abc
#a3 [1, 2, 3]
{% endhighlight %}

<p align="justify">
<b>How to inherit a parent class's __init__?</b><br>
Two ways:<br>
</p>
{% highlight Python %}
class Parent:
    def __init__(self, age=30):
        self.age = age

    def printAge(self):
        print('{} years old'.format(self.age))


class Child1(Parent):
    def __init__(self, age=10, name='Bob'):
        Parent.__init__(self, age=age)
        self.name = name

    def infos(self):
        print('{} is {} years old'.format(self.name, self.age))


class Child2(Parent):
    def __init__(self, age=5, name='Julie'):
        super(Child2, self).__init__(age=age)
        self.name = name

    def infos(self):
        print('{} is {} years old'.format(self.name, self.age))


if __name__ == '__main__':
    c1 = Child1()
    c1.infos()
    
    c2 = Child2()
    c2.infos()

#Output:
#Bob is 10 years old
#Julie is 5 years old
{% endhighlight %}

<p align="justify">
<b>How to merge 2 dicts? How to delete one key-value in a dict</b><br>
</p>
{% highlight Python %}
d1 = {0: '0', 1: '1'}
d2 = {2: '2', 3: '3'}
# Merge 2 dict
d1.update(d2)
# Delete one key-value
del d1[0]
{% endhighlight %}

<p align="justify">
<b>What is GIL?</b><a href="https://www.zhihu.com/question/25532384"> [1]</a><br>
GIL is Global Interpreter Lock. In a process, there might be more than 1 thread. If one thread is running, GIL orders the other threads wait until the running thread is finished. But it is possible that several process are running at a same time.<br>
Thread is a unit of CPU scheduling; Process is a unit of computing ressources. We can regard a process as a train, a thread as a wagon.<br>
A process has mutiple threads. $\rightarrow$ A train has mutilple wagons.<br>
Different process have different data. $\rightarrow$ Different trains have different passengers.<br>
At a same process, data is shared among different threads. $\rightarrow$ At a same train, one passenger can move from one wagon to another.<br>
One process cannot influence another, but if a thread breaks down, the process is over. $\rightarrow$ One train cannot disturb another train.
</p>
{% highlight Python %}

{% endhighlight %}

<p align="justify">
<b>What is decorator?</b><a href="https://foofish.net/python-decorator.html"> [2]</a><br>
</p>
{% highlight Python %}

{% endhighlight %}


<p align="justify">
<b>How ?</b><br>
</p>
{% highlight Python %}

{% endhighlight %}