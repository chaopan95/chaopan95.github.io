---
layout: post
title:  "Leetcode"
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


## 0001. Two Sum
<p align="justify">
Given an array of integers, return indices of the two numbers such that they add up to a specific target. You may assume that each input would have exactly one solution, and you may not use the same element twice.
</p>
{% highlight C++ %}
/*
Example
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
*/
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        int n = int(nums.size());
        if (n < 2) { return res; }
        unordered_map<int, int> M;
        unordered_map<int, int>::iterator iter;
        for (int i = 0; i < n; i++)
        {
            int diff = target - nums[i];
            iter = M.find(diff);
            if (iter != M.end())
            {
                res.push_back(i);
                res.push_back(iter->second);
                return res;
            }
            else { M[nums[i]] = i; }
        }
        return res;
    }
};
{% endhighlight %}

## 0002. Add Two Numbers
<p align="justify">
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.
</p>
{% highlight C++ %}
/*
Example:
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
*/
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *pre = new ListNode(0), *cur = pre;
        int val = 0, pos = 0;
        while (l1 != nullptr && l2 != nullptr)
        {
            val = l1->val + l2->val + pos;
            pos = 0;
            if (val > 9)
            {
                val -= 10;
                pos = 1;
            }
            cur->next = new ListNode(val);
            cur = cur->next;
            l1 = l1->next;
            l2 = l2->next;
        }
        while (l1 != nullptr)
        {
            val = l1->val + pos;
            pos = 0;
            if (val > 9)
            {
                val -= 10;
                pos = 1;
            }
            cur->next = new ListNode(val);
            cur = cur->next;
            l1 = l1->next;
        }
        while (l2 != nullptr)
        {
            val = l2->val + pos;
            pos = 0;
            if (val > 9)
            {
                val -= 10;
                pos = 1;
            }
            cur->next = new ListNode(val);
            cur = cur->next;
            l2 = l2->next;
        }
        if (pos == 1) { cur->next = new ListNode(1); }
        ListNode *head = pre->next;
        delete pre;
        return head;
    }
};
{% endhighlight %}

## 0003. Longest Substring Without Repeating Characters*
<p align="justify">
Given a string, find the length of the longest substring without repeating characters.<br><br>

<b>Example 1:</b><br>
Input: "abcabcbb"<br>
Output: 3 <br>
Explanation: The answer is "abc", with the length of 3. <br><br>

<b>Example 2:</b><br>
Input: "bbbbb"<br>
Output: 1<br>
Explanation: The answer is "b", with the length of 1.<br><br>

<b>Example 3:</b><br>
Input: "pwwkew"<br>
Output: 3<br>
Explanation: The answer is "wke", with the length of 3. <br>
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.<br><br>

<b>Solution:</b><br>
$\bigstar$ Dynamic programming<br>
We need a dictionary dict (Attention: dict is a key word in Python, take another) for each character to stock its last position in one string and we need a array dp for each character to represente a max length without repetition if we regard this character as an end.<br>
At first, if the length of s is 0, return directly 0.<br>
Then, dp[0] = 1 which means max length is 1 for the first character. We take i (i>=1) into account.<br>
If s[i] is not in dict, we regard s[i] is never visited, in otehr word, s[i] is different from any character from s[0] to s[i-1], so we have dp[i] = dp[i-1]+1.<br>
If s[i] is in dict, s[i] is repeated and we calculate i-dict[s[i]], which means a distance dist between two same s[i]. If dist > dp[i-1], a max length without repetition for s[i] is bigger than s[i-1], in other word, s[i-1] has one repetition before, then dp[i] = dp[i-1]+1; otherwise, dp[i] = dist.
</p>
{% highlight C++ %}
// Dynamic programming
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = int(s.length()), maxLen = 1;
        if (n == 0) { return 0; }
        map<int, int> dict;
        map<int, int>::iterator iter;
        int *dp = new int [n]{};
        dp[0] = 1;
        dict[s[0]] = 0;
        for (int i = 1; i < n; i++)
        {
            iter = dict.find(s[i]);
            if (iter == dict.end()) { dp[i] = dp[i-1] + 1; }
            else
            {
                int dist = i - dict[s[i]];
                if (dist > dp[i-1]) { dp[i] = dp[i-1] + 1; }
                else { dp[i] = dist; }
            }
            dict[s[i]] = i;
            maxLen = maxLen > dp[i] ? maxLen : dp[i];
        }
        delete []dp;
        return maxLen;
    }
};
{% endhighlight %}

<p align="justify">
$\bigstar$ Sliding window<br>
1) We initialise a dict for all characters of s with false, which means no one is visited. We set two index or pointers i, j for sliding window: i = j = 0 or i, j points to s[0].<br>
2) Under a loop of i < length(s) and j < length(s), if dict[s[i]] is false, set dict[s[i]] = true, max_len = max(max_len, i-j+1), i++, which means s[i] is never visited, we cans slide i further; otherwise, set dict[s[j]] = false, j++, which means s[i] and s[j] are same characters, we have to get rid of the influence of i, that is to say, s[i] is visited.
</p>
{% highlight C++ %}
// Sliding windows
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = int(s.length());
        if (n == 0) { return 0; }
        int i = 0, j = 0, maxLen = 1;
        map<int, bool> dict;
        for (int i = 0; i < n; i++) { dict[s[i]] = false; }
        while (i < n && j < n)
        {
            if (!dict[s[i]])
            {
                dict[s[i]] = true;
                maxLen = maxLen > (i-j+1) ? maxLen : (i-j+1);
                i++;
            }
            else
            {
                dict[s[j]] = false;
                j++;
            }
        }
        return maxLen;
    }
};
{% endhighlight %}

## 0004. Median of Two Sorted Arrays*
<p align="justify">
There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)). You may assume nums1 and nums2 cannot be both empty.<br><br>

<b>Example:</b><br>
nums1 = [1, 3]<br>
nums2 = [2]<br>
The median is 2.0<br><br>

nums1 = [1, 2]<br>
nums2 = [3, 4]<br>
The median is (2 + 3)/2 = 2.5<br><br>

<b>Solution:</b><br>
This problem requires a time complexity in <b>O(log(m+n))</b>, which determines that we cannot merge sort the twos arrays then get a median value because of o(m+n). A faisible solution is to find k-th value in two sorted arrays based on binary search.<br><br>

$\bigstar$ Cut two sorted arrays<br>
In fact, median is a statistique term, which means a value in middle position. If we want acquire a median value in one sorted array, it is easy to take the value in middlle position.
$$
\begin{matrix}
 & &  & & \text{cut} & &  & &  \\
 & &  & & \Downarrow & &  & &  \\
1 & & 2 & & 3 & & 4 & & 5 \\
 & &  & & \Uparrow & &  & &  \\
 & &  & & \text{cut} & &  & &
\end{matrix}
\quad \quad \quad \text{or} \quad \quad \quad
\begin{matrix}
 & &  & &  & \text{cut} &  & &  & & \\
 & &  & &  & \Downarrow &  & &  & & \\
1 & & 2 & & 3 & & 4 & & 5 & & 6\\
 & &  & &  & \Uparrow &  & &  & & \\
 & &  & &  & \text{cut} &  & &  & &
\end{matrix} \quad
$$

For example, the cutting line split an array into two parts with same size, l1 and r1 are left value and right value of cut postion.<br>
Left example is odd array, l1 = r1 = 3, while right example is even array, l1 = 3, r1 = 4.<br><br>
Similarly, we can generalize this to two arrays. If we can find such a cut for two arrays that the number of left cut and the number of right cut are equal, we can determine a median value for the 2 arrays. For example, we determine cut position k1, k2 for A, B respectively.
$$
\begin{matrix}
 & & & \text{cut} & & & \\
 & & & \Downarrow & & & \\
1 & & 3 & & 5 & & 7\\
\\
 & 2 & & 4 & & 6 \\
 & & & \Uparrow & & \\
 & & & \text{cut} & & &
\end{matrix}
\quad \quad
\begin{matrix}
l_{1} \text{ = 3, } r_{1} \text{ = 5} \\
\\
l_{2} \text{ = 4, } r_{2} \text{ = 4} \\
\end{matrix}
\Rightarrow
\text{Median} = \frac{\max(l_{1}, l_{2}) + \min(r_{1}, r_{2})}{2} = 4
$$

$\bigstar$ Unify odd and even number<br>
According to traditional way, we have to treat case-by-case: it is different to calculate median value for odd array and even array.<br>
In order to conquer this problem, we introuce a virtual placeholder. For example, an original array A is <b>1 2 3 4 5</b> and we insert a virtual placeholder <b>#</b> into A, which becomes A' = <b>#1#2#3#4#5#</b>. We see, lenghth of the original array is 5, while length of the new array is 5*2+1. If we cut A' at position 4(5th character from left). With this method, it's easy to get value in the original. Cut position c = 4
$$
\begin{aligned}
&
\begin{matrix}
 & & & & \text{cut} & & & & & \\
 & & & & \Downarrow & & & & & \\
\# & 1 & \# & 2 & \# & 3 & \# & 4 & \# & 5 & \# \\
 & & & & \Uparrow & & & & & \\
 & & & & \text{cut} & & & & &
\end{matrix} \\
& l_{1} = A[\frac{c-1}{2}] = A[1] = 2 \\
& r_{1} = A[\frac{c}{2}] = A[2] = 3
\end{aligned}
$$

If cut position for A' is in first # or last #, it will cause a overflow problem.
$$
\begin{matrix}
 & & & &  & & \text{cut} & & & & &  & \\
 & & & &  & & \Downarrow & & & & &  & \\
\# & 1 & \# & 2 & \#  & 3 & \# & & & & & & \\
\\
 & & & & & & \# & 4 & \# & 5 & \# & 6 & \# \\
 & & & &  & & \Uparrow & & & & &  & \\
 & & & &  & & \text{cut} & & & & &  &
\end{matrix}
\quad \quad
\begin{matrix}
l_{1} \text{ = 3, } r_{1} \text{ = 6} \\
\\
l_{2} \text{ = 1, } r_{2} \text{ = 4} \\
\end{matrix}
$$
As the example shows, if cut position at first #, left = min(A[0], B[0]), while cut position at last #, right = max(A[-1], B[-1]).
</p>
{% highlight C++ %}
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1 = int(nums1.size()), n2 = int(nums2.size());
        if (n1 == 0) { return (nums2[(n2-1)/2]+nums2[n2/2])/2.0; }
        if (n2 == 0) { return (nums1[(n1-1)/2]+nums1[n1/2])/2.0; }
        int MAX = nums1[n1-1] > nums2[n2-1] ? nums1[n1-1] : nums2[n2-1];
        int MIN = nums1[0] < nums2[0] ? nums1[0] : nums2[0];
        if (n1 < n2) { return findK(nums1, nums2, 0, 2*n1, n1, n2, MAX, MIN); }
        else { return findK(nums2, nums1, 0, 2*n2, n2, n1, MAX, MIN); }
    }
    double findK(vector<int> nums1, vector<int> nums2,
                 int b, int e, int n1, int n2,
                 int MAX, int MIN)
    {
        int l1 = 0, l2 = 0, r1 = 0, r2 = 0;
        int k1 = (b + e) / 2, k2 = (n1 + n2 - k1);
        if (k1 == 0) { l1 = MIN; r1 = nums1[0]; }
        else if (k1 >= 2*n1) { l1 = nums1[n1-1]; r1 = MAX; }
        else { l1 = nums1[(k1-1)/2]; r1 = nums1[k1/2]; }
        if (k2 == 0) { l2 = MIN; r2 = nums2[0]; }
        else if (k2 >= 2*n2) { l2 = nums2[n2-1]; r2 = MAX; }
        else { l2 = nums2[(k2-1)/2]; r2 = nums2[k2/2]; }
        if (l1 <= r2 && l2 <= r1)
        {
            return ((l1>l2?l1:l2)+(r1 < r2?r1:r2))/2.0;
        }
        else if (l1 > r2)
        {
            return findK(nums1, nums2, b, k1-1, n1, n2, MAX, MIN);
        }
        else { return findK(nums1, nums2, k1+1, e, n1, n2, MAX, MIN); }
    }
};
{% endhighlight %}

## 0005. Longest Palindromic Substring*
<p align="justify">
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.<br><br>

<b>Example 1:</b><br>
Input: "babad"<br>
Output: "bab"<br>
Note: "aba" is also a valid answer.<br><br>

<b>Example 2:</b><br>
Input: "cbbd"<br>
Output: "bb"<br><br>

<b>Solution:</b><br>
Dynamic programming:<br>
We construct a table, such that table[i][j] represent a substring fron s[i] to s[j] is palindromic or not. Note that, each single character is palindromic, namely table[i][i] is true. We also assignment table[i][i+1] = true if two adjacent characters are same. Then we visit other substirngs to find the longest one.
</p>
{% highlight C++ %}
class Solution {
public:
    string longestPalindrome(string s) {
        int n = int(s.length()), pos = 0, len = 1;
        if (n < 2) { return s; }
        bool **dp = new bool *[n];
        for (int i = 0; i < n; i++)
        {
            dp[i] = new bool [n]{};
            dp[i][i] = true;
        }
        for (int i = 0; i < n-1; i++)
        {
            if (s[i] == s[i+1])
            {
                dp[i][i+1] = true;
                pos = i;
                len = 2;
            }
        }
        for (int k = 2; k < n; k++)
        {
            for (int i = 0; i < n-k; i++)
            {
                int j = i + k;
                if (s[i] == s[j] && dp[i+1][j-1])
                {
                    pos = i;
                    len = k + 1;
                    dp[i][j] = true;
                }
            }
        }
        for (int i = 0; i < n; i++) { delete []dp[i]; }
        delete []dp;
        return s.substr(pos, len);
    }
};
{% endhighlight %}

## 0006. ZigZag Conversion
<p align="justify">
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)<br>
P   A   H   N<br>
A P L S I I G<br>
Y   I   R<br><br>

And then read line by line: "PAHNAPLSIIGYIR"<br><br>

Write the code that will take a string and make this conversion given a number of rows:<br>
string convert(string s, int numRows);<br><br>

<b>Example 1:</b><br>
Input: s = "PAYPALISHIRING", numRows = 3<br>
Output: "PAHNAPLSIIGYIR"<br><br>

<b>Example 2:</b><br>
Input: s = "PAYPALISHIRING", numRows = 4<br>
Output: "PINALSIGYAHRPI"<br>
Explanation:<br>
P     I    N<br>
A   L S  I G<br>
Y A   H R<br>
P     I<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    string convert(string s, int numRows) {
        int n = int(s.length());
        if (n == 0 || numRows < 2) { return s; }
        string res = "";
        int nRow = numRows, num = 2*numRows - 2;
        int nCol = (n / num) * (num - nRow + 1);
        if (n % num < nRow) { nCol++; }
        else { nCol += (n % num) - nRow + 1; }
        char *mat = new char [nRow*nCol]{};
        for (int k = 0; k < n; k++)
        {
            int quo = k / num, mod = k % num;
            int i = -1, j = quo * (num - nRow + 1);
            if (mod < nRow) { i = mod; }
            else
            {
                i = nRow - 2 - (mod - nRow);
                j += mod - nRow + 1;
            }
            mat[i*nCol+j] = s[k];
        }
        for (int i = 0; i < nRow; i++)
        {
            for (int j = 0; j < nCol; j++)
            {
                if (mat[i*nCol+j]) { res.push_back(mat[i*nCol+j]); }
            }
        }
        delete []mat;
        return res;
    }
};
{% endhighlight %}

## 0007. Reverse Integer
<p align="justify">
Given a 32-bit signed integer, reverse digits of an integer.<br><br>

<b>Example 1:</b><br>
Input: 123<br>
Output: 321<br><br>

<b>Example 2:</b><br>
Input: -123<br>
Output: -321<br><br>

<b>Example 3:</b><br>
Input: 120<br>
Output: 21<br><br>

Note:<br>
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [$-2^{31}$,  $2^{31}$−1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int reverse(int x) {
        long res = 0;
        bool isNeg = false;
        if (x < 0) { isNeg = true; }
        while (x)
        {
            res = res * 10 + x % 10;
            x /= 10;
        }
        if (res >= (1ll << 31) || res <= -(1ll << 31)) { return 0; }
        return int(res);
    }
};
{% endhighlight %}

## 0008. String to Integer (atoi)
<p align="justify">
Implement atoi which converts a string to an integer. The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.<br><br>

The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.<br><br>

If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.<br><br>

If no valid conversion could be performed, a zero value is returned.<br><br>

Note:<br>
Only the space character ' ' is considered as whitespace character.<br>
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [$-2^{31}$,  $2^{31}$−1]. If the numerical value is out of the range of representable values, INT_MAX ($2^{31}$ − 1) or INT_MIN (-$2^{31}$) is returned.<br><br>

<b>Example:</b><br>
Input: "42"<br>
Output: 42<br><br>

<b>Example 2:</b><br>
Input: "   -42"<br>
Output: -42<br>
Explanation: The first non-whitespace character is '-', which is the minus sign. Then take as many numerical digits as possible, which gets 42.<br><br>

<b>Example 3:</b><br>
Input: "4193 with words"<br>
Output: 4193<br>
Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.<br><br>

<b>Example 4:</b><br>
Input: "words and 987"<br>
Output: 0<br>
Explanation: The first non-whitespace character is 'w', which is not a numerical digit or a +/- sign. Therefore no valid conversion could be performed.<br><br>

<b>Example 5:</b><br>
Input: "-91283472332"<br>
Output: -2147483648<br>
Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer. Thefore INT_MIN (−$2^{31}$) is returned.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int myAtoi(string s) {
        int n = int(s.length());
        if (n == 0) { return 0; }
        bool isNeg = false;
        long res = 0;
        int pos = 0;
        while (s[pos] == ' ') { pos++; }
        if (s[pos] == '-') { isNeg = true; }
        else if (s[pos] == '+') { isNeg = false; }
        else if (s[pos] >= '0' && s[pos] <= '9') { res += s[pos] - '0'; }
        else { return 0; }
        for (++pos; pos < n; pos++)
        {
            if (s[pos] < '0' || s[pos] > '9') { break; }
            res = res * 10 + s[pos] - '0';
            if (res >= (1ll << 31))
            {
                if (isNeg) { return -(1ll << 31); }
                else { return (1ll << 31) - 1; }
            }
        }
        if (isNeg) { return -res; }
        return res;
    }
};
{% endhighlight %}

## 0009. Palindrome Number
<p align="justify">
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.<br><br>

<b>Example 1:</b><br>
Input: 121<br>
Output: true<br><br>

<b>Example 2:</b><br>
Input: -121<br>
Output: false<br>
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.<br><br>

<b>Example 3:</b><br>
Input: 10<br>
Output: false<br>
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.<br><br>

Follow up:<br>
Coud you solve it without converting the integer to a string?<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) { return false; }
        if (x < 10) { return true; }
        int count = 0, a = x;
        while (a)
        {
            count++;
            a /= 10;
        }
        count--;
        while (x)
        {
            int deno = int(pow(10, count));
            int left = x / deno, right = x % 10;
            if (left != right) { return false; }
            x = x % deno;
            x = x / 10;
            count -= 2;
        }
        return true;
    }
};
{% endhighlight %}

## 0010. Regular Expression Matching*
<p align="justify">
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.<br><br>

'.' Matches any single character.<br>
'*' Matches zero or more of the preceding element.<br>

The matching should cover the entire input string (not partial).<br><br>

Note:<br>
s could be empty and contains only lowercase letters a-z.<br>
p could be empty and contains only lowercase letters a-z, and characters like . or *.<br><br>

<b>Example 1:</b><br>
Input:<br>
s = "aa"<br>
p = "a"<br>
Output: false<br>
Explanation: "a" does not match the entire string "aa".<br><br>

<b>Example 2:</b><br>
Input:<br>
s = "aa"<br>
p = "a*"<br>
Output: true<br>
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".<br><br>

<b>Example 3:</b><br>
Input:<br>
s = "ab"<br>
p = ".*"<br>
Output: true<br>
Explanation: ".*" means "zero or more (*) of any character (.)".<br><br>

<b>Example: 4</b><br>
Input:<br>
s = "aab"<br>
p = "c*a*b"<br>
Output: true<br>
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".<br><br>

<b>Example 5:</b><br>
Input:<br>
s = "mississippi"<br>
p = "mis*is*p*."<br>
Output: false<br><br>

<b>Solution:</b><br>
Consider two string s, p. i, j denote some position in s and p.<br><br>

We use dynamic programming with a matrix dp $(ns+1) \times (np+1)$. ns and np are length of s and p. dp[i][j] denote if s[i:] and p[j:] match. The last entry dp[ns][np] is true. because '' and '' match.<br><br>

Firstly, we check if s[i] and p[j] match ij_match is false or true.<br>
Secondly, if p[j+1] is *, dp[i][j] has two independent sources: dp[i][j+2] and dp[i+1][j]. In detail, dp[i][j+2] represents we jump over * to check dp[i][j+2]. Similarly, dp[i+1][j] influences dp[i][j] but we have to consider ij_match. Both ij_match and dp[i+1][j] are true, dp[i][j] is true.<br>
Otherwise, p[j+1] is not *, dp[i][j] is dependent of ij_match and dp[i+1][j+1].<br>
Finally, return dp[0][0]<br><br>

Remark: why we don't consider ij_match for dp[i][j+2]? Because we jump over *.
</p>
{% highlight C++ %}
class Solution {
public:
    bool isMatch(string s, string p) {
        int ns = int(s.length()), np = int(p.length());
        bool **dp = new bool *[ns+1];
        for (int i = 0; i <= ns; i++) { dp[i] = new bool [np+1]{}; }
        dp[ns][np] = true;
        for (int i = ns; i >= 0; i--)
        {
            for (int j = np-1; j >= 0; j--)
            {
                bool ijMatch = i < ns && (s[i] == p[j] || p[j] == '.');
                if (j < np - 1 && p[j+1] == '*')
                {
                    dp[i][j] = dp[i][j+2] || (ijMatch && dp[i+1][j]);
                }
                else { dp[i][j] = ijMatch && dp[i+1][j+1]; }
            }
        }
        int res = dp[0][0];
        for (int i = 0; i <= ns; i++) { delete []dp[i]; }
        delete []dp;
        return res;
    }
};
{% endhighlight %}

## 0011. Container With Most Water*
<p align="justify">
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water. Note: You may not slant the container and n is at least 2.<br><br>

<b>Example:</b><br>
Input: [1,8,6,2,5,4,8,3,7]<br>
Output: 49<br><br>

<b>Solution:</b><br>
Brutal force cause Time Limit Exceeded. We use two points method.<br>
1) Firstly, we set two extremities as initial values.<br>
2) Then we move the shorter side towards the longer side one step, e.g. moving 1 towards 7.<br>
3) We repeat 2) until two side are adjacent.<br>
During this process, we calculate the max area.<br>
</p>
{% highlight C++ %}
/*
 	|	 	 	 	 	|	 	|
 	|	|	 	 	 	|	 	|
 	|	|	 	|	 	|	 	|
 	|	|	 	|	|	|	 	|
 	|	|	 	|	|	|	|	|
 	|	|	|	|	|	|	|	|
|	|	|	|	|	|	|	|	|
The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7].
In this case, the max area of water (blue section) the container can
contain is 49.
*/

class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = int(height.size());
        if (n < 2) { return 0; }
        int i = 0, j = n-1, res = 0;
        while (i < j)
        {
            int a = height[i], b = height[j];
            int B = a < b ? a : b, H = j - i;
            if (B * H > res) { res = B * H; }
            if (a < b) { i++; }
            else { j--; }
        }
        return res;
    }
};
{% endhighlight %}

## 0012. Integer to Roman
<p align="justify">
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.<br>
<table class="a">
  <tr><th>Symbol</th><th>Value</th></tr>
  <tr><td>I</td><td>1</td></tr>
  <tr><td>V</td><td>5</td></tr>
  <tr><td>X</td><td>10</td></tr>
  <tr><td>L</td><td>50</td></tr>
  <tr><td>C</td><td>100</td></tr>
  <tr><td>D</td><td>500</td></tr>
  <tr><td>M</td><td>1000</td></tr>
</table><br>
</p>
<p align="justify">
For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.<br><br>

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:<br><br>

I can be placed before V (5) and X (10) to make 4 and 9.<br>
X can be placed before L (50) and C (100) to make 40 and 90.<br>
C can be placed before D (500) and M (1000) to make 400 and 900.<br>
Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.<br><br>

<b>Example:</b><br>
Input: 3<br>
Output: "III"<br><br>

Input: 4<br>
Output: "IV"<br><br>

Input: 9<br>
Output: "IX"<br><br>

Input: 58<br>
Output: "LVIII"<br>
Explanation: L = 50, V = 5, III = 3.<br><br>

Input: 1994<br>
Output: "MCMXCIV"<br>
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    string intToRoman(int num) {
        string res = "";
        int nums[4] = {1000, 100, 10, 1};
        char roman[4] = {'M', 'C', 'X', 'I'};
        for (int i = 0; i < 4; i++)
        {
            int quo = num / nums[i], mod = num % nums[i];
            num = mod;
            if (quo == 0) { continue; }
            if (quo == 9)
            {
                if (roman[i] == 'C') { res += "CM"; }
                if (roman[i] == 'X') { res += "XC"; }
                if (roman[i] == 'I') { res += "IX"; }
                continue;
            }
            if (quo == 4)
            {
                if (roman[i] == 'C') { res += "CD"; }
                if (roman[i] == 'X') { res += "XL"; }
                if (roman[i] == 'I') { res += "IV"; }
                continue;
            }
            if (quo >= 5)
            {
                quo -= 5;
                if (roman[i] == 'C') { res.push_back('D'); }
                if (roman[i] == 'X') { res.push_back('L'); }
                if (roman[i] == 'I') { res.push_back('V'); }
            }
            while (quo--) { res.push_back(roman[i]); }
        }
        return res;
    }
};
{% endhighlight %}

## 0013. Roman to Integer
<p align="justify">
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.<br>
<table class="a">
  <tr><th>Symbol</th><th>Value</th></tr>
  <tr><td>I</td><td>1</td></tr>
  <tr><td>V</td><td>5</td></tr>
  <tr><td>X</td><td>10</td></tr>
  <tr><td>L</td><td>50</td></tr>
  <tr><td>C</td><td>100</td></tr>
  <tr><td>D</td><td>500</td></tr>
  <tr><td>M</td><td>1000</td></tr>
</table><br>
</p>
<p align="justify">
For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.<br><br>

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:<br><br>

I can be placed before V (5) and X (10) to make 4 and 9.<br>
X can be placed before L (50) and C (100) to make 40 and 90.<br>
C can be placed before D (500) and M (1000) to make 400 and 900.<br><br>

Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.<br><br>

<b>Example 1:</b><br>
Input: "III"<br>
Output: 3<br><br>

<b>Example 2:</b><br>
Input: "IV"<br>
Output: 4<br><br>

<b>Example 3:</b><br>
Input: "IX"<br>
Output: 9<br><br>

<b>Example 4:</b><br>
Input: "LVIII"<br>
Output: 58<br>
Explanation: L = 50, V= 5, III = 3.<br><br>

<b>Example 5:</b><br>
Input: "MCMXCIV"<br>
Output: 1994<br>
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int romanToInt(string s) {
        int res = 0, n = int(s.length());
        if (n == 0) { return res; }
        int nums[4] = {1000, 100, 10, 1};
        char roman[4] = {'M', 'C', 'X', 'I'};
        map<char, int> dictRomanToInt;
        dictRomanToInt['M'] = 1000;
        dictRomanToInt['D'] = 500;
        dictRomanToInt['C'] = 100;
        dictRomanToInt['L'] = 50;
        dictRomanToInt['X'] = 10;
        dictRomanToInt['V'] = 5;
        dictRomanToInt['I'] = 1;
        map<string, int> dictRomanpairToInt;
        map<string, int>::iterator iter;
        dictRomanpairToInt["IV"] = 4;
        dictRomanpairToInt["IX"] = 9;
        dictRomanpairToInt["XL"] = 40;
        dictRomanpairToInt["XC"] = 90;
        dictRomanpairToInt["CD"] = 400;
        dictRomanpairToInt["CM"] = 900;
        int pos = 0;
        while (pos < n)
        {
            iter = dictRomanpairToInt.find(s.substr(pos, 2));
            if (iter != dictRomanpairToInt.end())
            {
                res += iter->second;
                pos += 2;
                continue;
            }
            res += dictRomanToInt[s[pos]];
            pos++;
        }
        return res;
    }
};
{% endhighlight %}

## 0014. Longest Common Prefix
<p align="justify">
Write a function to find the longest common prefix string amongst an array of strings.<br><br>

If there is no common prefix, return an empty string "".<br><br>

<b>Example 1:</b><br>
Input: ["flower","flow","flight"]<br>
Output: "fl"<br><br>

<b>Example 2:</b><br>
Input: ["dog","racecar","car"]<br>
Output: ""<br>
Explanation: There is no common prefix among the input strings.<br><br>

<b>Note:</b><br>
All given inputs are in lowercase letters a-z.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int n = int(strs.size());
        if (n == 0) { return ""; }
        string str = "";
        for (int j = 0; j < int(strs[0].length()); j++)
        {
            char flag = strs[0][j];
            for (int i = 1; i < n; i++)
            {
                if (strs[i][j] != flag) { return str; }
            }
            str.push_back(flag);
        }
        return str;
    }
};
{% endhighlight %}

## 0015. 3Sum*
<p align="justify">
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero. Note: The solution set must not contain duplicate triplets.<br><br>

<b>Solution:</b>
$$
\begin{matrix}
-4 & -1 & -1 & 0 & 1 & 2 \\
\Uparrow & \Uparrow &  &  &  & \Uparrow \\
i & j &   &  &  & k
\end{matrix}
$$

1) sort the array<br>
2) set 3 position variable i, j, k: i is from 0 to n-3, j is initialised as i+1 and k is n-1 initially.<br>
3) compute sum of nums for i, j, k. if sum < 0, j++; if sum > 0, k--; if sum is 0, append this tuple into our resuslt array. Conitnue to move j rightward until no repetiton and continue to move k until no duplicate.<br>
4) repeat 3) until i ends<br>
</p>
{% highlight C++ %}
/*
Given array nums = [-1, 0, 1, 2, -1, -4],
A solution set is:
[[-1, 0, 1],[-1, -1, 2]]
*/
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = int(nums.size());
        vector<vector<int>> ans;
        if (n < 3) { return ans; }
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n - 2; i++)
        {
            int j = i + 1, k = n - 1;
            while (j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < 0) { j++; }
                else if (sum > 0) { k--; }
                else
                {
                    while (i < k && nums[i] == nums[i+1]) { i++; }
                    while (j < k && nums[j] == nums[j+1]) { j++; }
                    while (j < k && nums[k] == nums[k-1]) { k--; }
                    ans.push_back({nums[i], nums[j], nums[k]});
                    j++;
                    k--;
                }
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0016. 3Sum Closest
<p align="justify">
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
</p>
{% highlight C++ %}
/*
Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation:
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
*/
class Solution {
public:
    void swap(vector<int>& nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    void quickSort(vector<int>& nums, int start, int end){
        if (start < end)
        {
            int i = start, j = end;
            while (i < j)
            {
                while (nums[i] <= nums[j] && i < j) { j--; }
                swap(nums, i, j);
                while (nums[i] <= nums[j] && i < j) { i++; }
                swap(nums, i, j);
            }
            quickSort(nums, start, i-1);
            quickSort(nums, j+1, end);
        }
    }
    int threeSumClosest(vector<int>& nums, int target) {
        int n = int(nums.size());
        quickSort(nums, 0, n-1);
        int sum = nums[0] + nums[1] + nums[2];
        int res = sum;
        for (int i = 0; i < n - 2; i++)
        {
            int j = i + 1, k = n - 1;
            while (j < k)
            {
                sum = nums[i] + nums[j] + nums[k];
                if (sum == target) { return sum; }
                if (abs(sum - target) < abs(res - target)) { res = sum; }
                if (sum < target) { j++; }
                else { k--; }
            }
        }
        return res;
    }
};
{% endhighlight %}

{% highlight C++ %}
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int n = int(nums.size()), res = 0, diff = (1ll << 31) - 1;
        if (n < 3) { return 0; }
        sort(nums.begin(), nums.end());
        for (int i = 0; i <= n-3; i++)
        {
            int j = i + 1, k = n - 1;
            while (j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];
                if (abs(sum - target) < diff)
                {
                    diff = abs(sum - target);
                    res = sum;
                }
                if (sum < target) { j++; }
                else if (sum > target) { k--; }
                else { return target; }
            }
        }
        return res;
    }
};
{% endhighlight %}

## 0017. Letter Combinations of a Phone Number
<p align="justify">
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.<br><br>

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
$$
\begin{bmatrix}
\text{1: } \infty & \text{2: abc} & \text{3: def} \\
4: ghi & 5: jkl & 6: mno \\
7: pqrs & 8: tuv & 9: wxyz
\end{bmatrix}
$$

<b>Example:</b><br>
Input: "23"<br>
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].<br><br>

Note: Although the above answer is in lexicographical order, your answer could be in any order you want.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        int n = int(digits.length());
        if (n == 0) { return res; }
        string *arr = new string [10]{};
        for (int i = 2; i <= 6; i++)
        {
            string str = "";
            for (int j = 0; j < 3; j++) { str.push_back('a' + (i-2)*3+j); }
            arr[i] = str;
        }
        arr[7] = "pqrs";
        arr[8] = "tuv";
        arr[9] = "wxyz";
        string str = digits;
        combine(str, res, 0, n, arr, digits);
        delete []arr;
        return res;
    }
    void combine(string &str, vector<string> &res, int idx, int n,
                 string *arr, string digits)
    {
        if (idx == n) { res.push_back(str); }
        else
        {
            for (int i = 0; i < int(arr[digits[idx]-'0'].length()); i++)
            {
                str[idx] = arr[digits[idx]-'0'][i];
                combine(str, res, idx+1, n, arr, digits);
            }
        }
    }
};
{% endhighlight %}

## 0018. 4Sum
<p align="justify">
Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target. Note: The solution set must not contain duplicate quadruplets.
</p>
{% highlight C++ %}
/*
Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.
A solution set is:
[[-1,  0, 0, 1], [-2, -1, 1, 2], [-2,  0, 0, 2]]
*/
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int n = int(nums.size());
        vector<vector<int>> res;
        if (n < 4) { return res; }
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n-3; i++)
        {
            for (int j = i+1; j < n-2; j++)
            {
                int b = j + 1, e = n-1;
                while (b < e)
                {
                    int sum = nums[i] + nums[j] + nums[b] + nums[e];
                    if (sum < target) { b++; }
                    else if (sum > target) { e--; }
                    else
                    {
                        while (i < n-3 && nums[i] == nums[i+1]) { i++; }
                        while (j < n-2 && nums[j] == nums[j+1]) { j++; }
                        while (b < e && nums[b] == nums[b+1]) { b++; }
                        while (b < e && nums[e] == nums[e-1]) { e--; }
                        vector<int> arr;
                        arr.push_back(nums[i]);
                        arr.push_back(nums[j]);
                        arr.push_back(nums[b]);
                        arr.push_back(nums[e]);
                        res.push_back(arr);
                        b++;
                        e--;
                    }
                }
            }
        }
        return res;
    }
};
{% endhighlight %}

## 0019. Remove Nth Node From End of List*
{% highlight C++ %}
/*
Given linked list: 1->2->3->4->5, and n = 2.
After removing the second node from the end, the linked
list becomes 1->2->3->5.
*/
// Two pointers
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if (n < 1 || head == nullptr) { return head; }
        ListNode *pre = new ListNode(0, head);
        ListNode *p1 = pre, *p2 = head, *p3 = head;
        while (n--)
        {
            if (p3 == nullptr) { return head; }
            p3 = p3->next;
        }
        while (p3 != nullptr)
        {
            p3 = p3->next;
            p2 = p2->next;
            p1 = p1->next;
        }
        p1->next = p2->next;
        delete p2;
        head = pre->next;
        delete pre;
        return head;
    }
};
{% endhighlight %}

## 0020. Valid Parentheses
{% highlight C++ %}
/*
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

Input: s = "([)]"
Output: false

Input: s = "{[]}"
Output: true
*/
class Solution {
public:
    bool isValid(string s) {
        int n = int(s.length());
        stack<char> st;
        for (int i = 0; i < n; i++)
        {
            if (s[i] == '[' || s[i] == '{' || s[i] == '(')
            {
                st.push(s[i]);
            }
            else if ((s[i] == ')' && !st.empty() && st.top() == '(') ||
                     (s[i] == ']' && !st.empty() && st.top() == '[') ||
                     (s[i] == '}' && !st.empty() && st.top() == '{'))
            {
                st.pop();
            }
            else { return false; }
        }
        return st.empty();
    }
};
{% endhighlight %}

## 0021. Merge Two Sorted Lists
{% highlight C++ %}
/*
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
*/
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr) { return l2; }
        if (l2 == nullptr) { return l1; }
        ListNode *head = new ListNode(0);
        ListNode *curNode = head;
        while (l1 != nullptr && l2 != nullptr)
        {
            if (l1->val < l2->val)
            {
                curNode->next = l1;
                l1 = l1->next;
            }
            else
            {
                curNode->next = l2;
                l2 = l2->next;
            }
            curNode = curNode->next;
        }
        if (l1 != nullptr) { curNode->next = l1; }
        if (l2 != nullptr) { curNode->next = l2; }
        curNode = head->next;
        delete head;
        return curNode;
    }
};
{% endhighlight %}

## 0022. Generate Parentheses
{% highlight C++ %}
/*
For example, given n = 3, a solution set is:
[
"((()))",
"(()())",
"(())()",
"()(())",
"()()()"
]
*/
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        if (n == 0) { return res; }
        string str = "";
        append(res, str, n, 0, 0, 0);
        return res;
    }
    void append(vector<string> &res, string str, int n,
                int idx, int nLeft, int nRight)
    {
        if (idx == 2*n)
        {
            if (nRight == nLeft) { res.push_back(str); }
            return;
        }
        if (nLeft >= nRight && nLeft < n)
        {
            append(res, str+'(', n, idx+1, nLeft+1, nRight);
        }
        append(res, str+')', n, idx+1, nLeft, nRight+1);
    }
};
{% endhighlight %}

## 0023. Merge k Sorted Lists
{% highlight C++ %}
/*
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
*/
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int n = int(lists.size());
        if (n == 0) { return nullptr; }
        ListNode *pre = new ListNode(0), *head = pre, *cur = pre;
        while (true)
        {
            int idx = -1, min = (1ll << 31) - 1;
            for (int i = 0; i < n; i++)
            {
                if (lists[i] == nullptr) { continue; }
                if (lists[i]->val < min)
                {
                    idx = i;
                    min = lists[i]->val;
                }
            }
            if (idx == -1) { break; }
            cur->next = lists[idx];
            lists[idx] = lists[idx]->next;
            cur = cur->next;
        }
        head = pre->next;
        delete pre;
        return head;
    }
};
{% endhighlight %}

## 0024. Swap Nodes in Pairs
{% highlight C++ %}
/*
Given 1->2->3->4, you should return the list as 2->1->4->3.
*/
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) { return head; }
        ListNode *pre = new ListNode(0);
        ListNode *par = head, *son = head->next, *gra = pre;
        pre->next = head;
        while (son != nullptr)
        {
            par->next = son->next;
            son->next = par;
            gra->next = son;
            gra = par;
            par = par->next;
            if (par == nullptr) { break; }
            son = par->next;
        }
        head = pre->next;
        delete pre;
        return head;
    }
};
{% endhighlight %}

## 0025. Reverse Nodes in k-Group
{% highlight C++ %}
/*
Given this linked list: 1->2->3->4->5
For k = 2, you should return: 2->1->4->3->5
For k = 3, you should return: 3->2->1->4->5
*/
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (head == nullptr || head->next == nullptr ||
            k < 2) { return head; }
        ListNode *front = head;
        for (int i = 0; i < k; i++)
        {
            if (front == nullptr) { return head; }
            front = front->next;
        }
        bool isEnd = false;
        ListNode *pre = new ListNode(0, head);
        ListNode *gra = pre, *par = head, *son = head->next;
        while (true)
        {
            ListNode *temp = front, *p1 = gra, *p2 = par, *p3 = son;
            while (p2 != temp)
            {
                if (front == nullptr) { isEnd = true; }
                else { front = front->next; }
                p2->next = p1;
                p1 = p2;
                p2 = p3;
                if (p3 == nullptr) { break; }
                p3 = p3->next;
            }
            par->next = p2;
            gra->next = p1;
            gra = par;
            par = p2;
            son = p3;
            if (isEnd || p3 == nullptr) { break; }
        }
        head = pre->next;
        delete pre;
        return head;
    }
};
{% endhighlight %}

## 0026. Remove Duplicates from Sorted Array
{% highlight C++ %}
/*
Given nums = [0,0,1,1,1,2,2,3,3,4]
Your function should return length = 5
*/
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = int(nums.size());
        if (n < 2) { return n; }
        int len = 1;
        for (int i = 1; i < n; i++)
        {
            if (nums[i] == nums[i-1]) { continue; }
            nums[len++] = nums[i];
        }
        return len;
    }
};
{% endhighlight %}

## 0027. Remove Element
{% highlight C++ %}
/*
Given nums = [0,1,2,2,3,0,4,2], val = 2,
Your function should return length = 5, 
*/
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int n = int(nums.size()), len = 0;
        if (n == 0) { return 0; }
        for (int i = 0; i < n; i++)
        {
            if (nums[i] != val) { nums[len++] = nums[i]; }
        }
        return len;
    }
};
{% endhighlight %}

## 0028. Implement strStr()*
{% highlight C++ %}
/*
Input: haystack = "hello", needle = "ll"
Output: 2
*/
class Solution {
public:
    int strStr(string haystack, string needle) {
        string T = haystack, P = needle;
        int nT = int(T.length()), nP = int(P.length());
        if (nP == 0) { return 0; }
        if (nT == 0) { return -1; }
        int *next = new int [nP]{};
        int i = 0, j = 0, t = next[0] = -1, k = 0;
        while (k < nP-1)
        {
            if (t < 0 || P[t] == P[k]) { next[++k] = ++t; }
            else { t = next[t]; }
        }
        while (i < nT && j < nP)
        {
            if (j < 0 || T[i] == P[j]) { i++; j++; }
            else { j = next[j]; }
        }
        delete []next;
        if (i-j <= nT-nP) { return i - j; }
        else { return -1; }
    }
};
{% endhighlight %}

## 0029. Divide Two Integers*
<p align="justify">
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.<br>
Return the quotient after dividing dividend by divisor.<br>
The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.<br><br>

<b>Example 1:</b><br>
Input: dividend = 10, divisor = 3<br>
Output: 3<br>
Explanation: 10/3 = truncate(3.33333..) = 3.<br>
<b>Example 2:</b><br>
Input: dividend = 7, divisor = -3<br>
Output: -2<br>
Explanation: 7/-3 = truncate(-2.33333..) = -2.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int divide(int dividend, int divisor) {
        int MAX = (1ll << 31) - 1;
        long m = abs(long(dividend)), n = abs(long(divisor)), quo = 0;
        while (m >= n)
        {
            long s = n, incre = 1;
            while ((s << 1) <= m) { s <<= 1; incre <<= 1; }
            quo += incre;
            m -= s;
        }
        if ((dividend < 0) ^ (divisor < 0)) { quo *= -1; }
        return quo > MAX ? MAX : int(quo);
    }
};
{% endhighlight %}

## 0030. Substring with Concatenation of All Words*
<p align="justify">
You are given a string s and an array of strings words of the same length. Return all starting indices of substring(s) in s that is a concatenation of each word in words exactly once, in any order, and without any intervening characters.<br>
You can return the answer in any order.<br><br>

<b>Example:</b><br>
Input: s = "barfoothefoobarman", words = ["foo","bar"]<br>
Output: [0,9]<br>
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.<br>
The output order does not matter, returning [9,0] is fine too.<br>

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]<br>
Output: []<br>

Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]<br>
Output: [6,9,12]<br><br>

<b>Constraints:</b><br>
1 <= s.length <= $10^{4}$<br>
s consists of lower-case English letters.<br>
1 <= words.length <= 5000<br>
1 <= words[i].length <= 30<br>
words[i] consists of lower-case English letters.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> res;
        int n = int(words.size()), sLen = int(s.length());
        if (n == 0 || n * words[0].length() > sLen) { return res; }
        sort(words.begin(), words.end());
        int i = 0, wordLen = int(words[0].length()), wordsLen = n * wordLen;
        while (i + wordsLen <= sLen)
        {
            string subStr = s.substr(i, wordsLen);
            vector<string> tmp;
            for (int j = 0; j < wordsLen; j += wordLen)
            {
                tmp.push_back(subStr.substr(j, wordLen));
            }
            sort(tmp.begin(), tmp.end());
            bool isOk = true;
            for (int k = 0; k < n; k++)
            {
                if (tmp[k] != words[k])
                {
                    isOk = false;
                    break;
                }
            }
            if (isOk) { res.push_back(i); }
            i++;
        }
        return res;
    }
};
{% endhighlight %}

## 0031. Next Permutation*
<p align="justify">
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.<br>
If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).<br>
The replacement must be in-place and use only constant extra memory.<br>
Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.<br><br>

<b>Example:</b><br>
1,2,3 → 1,3,2<br>
3,2,1 → 1,2,3<br>
1,1,5 → 1,5,1<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = int(nums.size()), idx = -1;
        if (n < 2) { return; }
        for (int i = n - 1; i > 0; i--)
        {
            if (nums[i-1] < nums[i])
            {
                idx = i - 1;
                break;
            }
        }
        if (idx == -1)
        {
            sort(nums.begin(), nums.end());
            return;
        }
        for (int i = n - 1; i > idx; i--)
        {
            if (nums[i] > nums[idx])
            {
                swap(nums[i], nums[idx]);
                sort(nums.begin()+idx+1, nums.end());
                return;
            }
        }
    }
};
{% endhighlight %}

## 0032. Longest Valid Parentheses*
<p align="justify">
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.<br><br>

<b>Example 1:</b><br>
Input: "(()"<br>
Output: 2<br>
Explanation: The longest valid parentheses substring is "()"<br>
<b>Example 2:</b><br>
Input: ")()())"<br>
Output: 4<br>
Explanation: The longest valid parentheses substring is "()()"<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxLen = 0, n = int(s.length()), leftCount = 0;
        if (n < 2) { return maxLen; }
        int *dp = new int [n]{};
        for (int i = 0; i < n; i++)
        {
            if (leftCount == 0 && s[i] == ')') { continue; }
            if (s[i] == '(') { leftCount++; }
            else
            {
                leftCount--;
                dp[i] = dp[i-1] + 1;
                if (i - dp[i] * 2 >= 0) { dp[i] += dp[i-dp[i]*2]; }
            }
            maxLen = maxLen > dp[i] ? maxLen : dp[i];
        }
        delete []dp;
        return maxLen * 2;
    }
};
{% endhighlight %}

## 0033. Search in Rotated Sorted Array
<p align="justify">
You are given an integer array nums sorted in ascending order, and an integer target. Suppose that nums is rotated at some pivot unknown to you beforehand (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]). If target is found in the array return its index, otherwise, return -1.<br><br>

<b>Example:</b><br>
Input: nums = [4,5,6,7,0,1,2], target = 0<br>
Output: 4<br><br>

Input: nums = [4,5,6,7,0,1,2], target = 3<br>
Output: -1<br><br>

Input: nums = [1], target = 0<br>
Output: -1<br><br>

<b>Solution:</b><br>
</p>
{% highlight C++ %}
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int n = int(nums.size());
        if (n == 0) { return -1; }
        return binSear(nums, target, 0, n-1);
    }
    int binSear(vector<int> nums, int target, int b, int e)
    {
        if (b > e) { return -1; }
        if (nums[b] == target) { return b; }
        if (nums[e] == target) { return e; }
        int m = (b + e) / 2;
        if (nums[b] < nums[e])  //sorted array
        {
            if (nums[m] > target) { return binSear(nums, target, b, m-1); }
            else if (nums[m] == target) { return m; }
            else { return binSear(nums, target, m+1, e); }
        }
        else  //rotated array
        {
            if (nums[m] < nums[e] && target > nums[b])
            {
                return binSear(nums, target, b, m-1);
            }
            else if (nums[m] > nums[b] && target < nums[e])
            {
                return binSear(nums, target, m+1, e);
            }
            else
            {
                if (nums[m] < target) { return binSear(nums, target, m+1, e); }
                else if (nums[m] == target) { return m; }
                else { return binSear(nums, target, b, m-1); }
            }
        }
    }
};
{% endhighlight %}

## 0034. Find First and Last Position of Element in Sorted Array
<p align="justify">
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value. If target is not found in the array, return [-1, -1]. Follow up: Could you write an algorithm with O(log n) runtime complexity?<br><br>

<b>Example:</b><br>
Input: nums = [5,7,7,8,8,10], target = 8<br>
Output: [3,4]<br><br>

Input: nums = [5,7,7,8,8,10], target = 6<br>
Output: [-1,-1]<br><br>

Input: nums = [], target = 0<br>
Output: [-1,-1]<br><br>

<b>Constraints:</b><br>
0 <= nums.length <= $10^{5}$<br>
-$10^{9}$ <= nums[i] <= $10^{9}$<br>
nums is a non-decreasing array.<br>
-$10^{9}$ <= target <= $10^{9}$<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int n = int(nums.size());
        vector<int> res;
        res.push_back(-1);
        res.push_back(-1);
        if (n < 1) { return res; }
        res[0] = binSearFir(nums, target, 0, n-1);
        res[1] = binSearSec(nums, target, 0, n);
        return res;
    }
    int binSearFir(vector<int> &nums, int target, int b, int e)
    {
        if (b == e)
        {
            if (nums[b] == target) { return b; }
            else { return -1; }
        }
        int m = (b + e) / 2;
        if (nums[m] < target) { return binSearFir(nums, target, m+1, e); }
        else { return binSearFir(nums, target, b, m); }
    }
    int binSearSec(vector<int> &nums, int target, int b, int e)
    {
        if (b + 1 == e)
        {
            if (nums[b] == target) { return b; }
            else { return -1; }
        }
        int m = (b + e) / 2;
        if (nums[m] <= target) { return binSearSec(nums, target, m, e); }
        else { return binSearSec(nums, target, b, m); }
    }
};
{% endhighlight %}

## 0035. Search Insert Position
<p align="justify">
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.<br><br>

<b>Example:</b><br>
Input: nums = [1,3,5,6], target = 5<br>
Output: 2<br><br>

Input: nums = [1,3,5,6], target = 2<br>
Output: 1<br><br>

Input: nums = [1,3,5,6], target = 7<br>
Output: 4<br><br>

Input: nums = [1,3,5,6], target = 0<br>
Output: 0<br><br>

Input: nums = [1], target = 0<br>
Output: 0<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = int(nums.size());
        if (n == 0) { return 0; }
        return binSear(nums, 0, n-1, target);;
    }
    int binSear(vector<int> arr, int b, int e, int target)
    {
        if (target > arr[e]) { return e+1; }
        if (target < arr[b]) { return b; }
        int m = (b + e) / 2;
        if (arr[m] == target) { return m; }
        else if (arr[m] < target) { return binSear(arr, m+1, e, target); }
        else { return binSear(arr, b, m-1, target); }
    }
};

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = int(nums.size());
        if (n == 0) { return 0; }
        int b = 0, e = n - 1;
        while (b <= e)
        {
            int m = (b + e) / 2;
            if (nums[b] >= target) { return b; }
            if (nums[e] == target) { return e; }
            if (nums[m] > target) { e = m - 1; }
            else if (nums[m] < target) { b = m + 1; }
            else { return m; }
        }
        if (nums[e] < target) { return e + 1; }
        else { return e; }
    }
};
{% endhighlight %}

## 0036. Valid Sudoku*
<p align="justify">
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:<br>
Each row must contain the digits 1-9 without repetition.<br>
Each column must contain the digits 1-9 without repetition.<br>
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.<br>
Note:<br>
A Sudoku board (partially filled) could be valid but is not necessarily solvable.<br>
Only the filled cells need to be validated according to the mentioned rules.
$$
\begin{bmatrix}
5 & 3 & . & . & 7 & . & . & . & . \\
6 & . & . & 1 & 9 & 5 & . & . & . \\
. & 9 & 8 & . & . & . & . & 6 & . \\
8 & . & . & . & 6 & . & . & . & 3 \\
4 & . & . & 8 & . & 3 & . & . & 1 \\
7 & . & . & . & 2 & . & . & . & 6 \\
. & 6 & . & . & . & . & 2 & 8 & . \\
. & . & . & 4 & 1 & 9 & . & . & 5 \\
. & . & . & . & 8 & . & . & 7 & 9
\end{bmatrix}
$$

<b>Example:</b><br>
Input: board = <br>
[["5","3",".",".","7",".",".",".","."]<br>
,["6",".",".","1","9","5",".",".","."]<br>
,[".","9","8",".",".",".",".","6","."]<br>
,["8",".",".",".","6",".",".",".","3"]<br>
,["4",".",".","8",".","3",".",".","1"]<br>
,["7",".",".",".","2",".",".",".","6"]<br>
,[".","6",".",".",".",".","2","8","."]<br>
,[".",".",".","4","1","9",".",".","5"]<br>
,[".",".",".",".","8",".",".","7","9"]]<br>
Output: true<br><br>

Input: board = <br>
[["8","3",".",".","7",".",".",".","."]<br>
,["6",".",".","1","9","5",".",".","."]<br>
,[".","9","8",".",".",".",".","6","."]<br>
,["8",".",".",".","6",".",".",".","3"]<br>
,["4",".",".","8",".","3",".",".","1"]<br>
,["7",".",".",".","2",".",".",".","6"]<br>
,[".","6",".",".",".",".","2","8","."]<br>
,[".",".",".","4","1","9",".",".","5"]<br>
,[".",".",".",".","8",".",".","7","9"]]<br>
Output: false<br>
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.<br><br>

<b>Constraints:</b><br>
board.length == 9<br>
board[i].length == 9<br>
board[i][j] is a digit or '.'.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] == '.') { continue; }
                if (!isValidForij(board, i, j)) { return false; }
            }
        }
        return true;
    }
    bool isValidForij(vector<vector<char>> board, int i, int j)
    {
        int dupRow = 0, dupCol = 0;
        for (int k = 0; k < 9; k++)
        {
            if (board[k][j] == board[i][j]) { dupRow++; }
            if (board[i][k] == board[i][j]) { dupCol++; }
        }
        if (dupRow > 1 || dupCol > 1) { return false; }
        int m = i / 3, n = j / 3, dupSubBox = 0;
        for (int x = 0; x < 3; x++)
        {
            for (int y = 0; y < 3; y++)
            {
                if (board[m*3+x][n*3+y] == board[i][j]) { dupSubBox++; }
            }
        }
        if (dupSubBox > 1) { return false; }
        return true;
    }
};
{% endhighlight %}

## 0037. Sudoku Solver*
<p align="justify">
Write a program to solve a Sudoku puzzle by filling the empty cells. A sudoku solution must satisfy all of the following rules:<br>
Each of the digits 1-9 must occur exactly once in each row.<br>
Each of the digits 1-9 must occur exactly once in each column.<br>
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.<br>
The '.' character indicates empty cells.
$$
\begin{bmatrix}
5 & 3 & . & . & 7 & . & . & . & . \\
6 & . & . & 1 & 9 & 5 & . & . & . \\
. & 9 & 8 & . & . & . & . & 6 & . \\
8 & . & . & . & 6 & . & . & . & 3 \\
4 & . & . & 8 & . & 3 & . & . & 1 \\
7 & . & . & . & 2 & . & . & . & 6 \\
. & 6 & . & . & . & . & 2 & 8 & . \\
. & . & . & 4 & 1 & 9 & . & . & 5 \\
. & . & . & . & 8 & . & . & 7 & 9
\end{bmatrix} \Rightarrow
\begin{bmatrix}
5 & 3 & {\color{Red} 4} & {\color{Red} 6} & 7 & {\color{Red} 8} & {\color{Red} 9} & {\color{Red} 1} & {\color{Red} 2} \\
6 & {\color{Red} 7} & {\color{Red} 2} & 1 & 9 & 5 & {\color{Red} 3} & {\color{Red} 4} & {\color{Red} 8} \\
{\color{Red} 1} & 9 & 8 & {\color{Red} 3} & {\color{Red} 4} & {\color{Red} 2} & {\color{Red} 5} & 6 & {\color{Red} 7} \\
8 & {\color{Red} 5} & {\color{Red} 9} & {\color{Red} 7} & 6 & {\color{Red} 1} & {\color{Red} 4} & {\color{Red} 2} & 3 \\
4 & {\color{Red} 2} & {\color{Red} 6} & 8 & {\color{Red} 5} & 3 & {\color{Red} 7} & {\color{Red} 9} & 1 \\
7 & {\color{Red} 1} & {\color{Red} 3} & {\color{Red} 9} & 2 & {\color{Red} 4} & {\color{Red} 8} & {\color{Red} 5} & 6 \\
{\color{Red} 9} & 6 & {\color{Red} 1} & {\color{Red} 5} & {\color{Red} 3} & {\color{Red} 7} & 2 & 8 & {\color{Red} 4} \\
{\color{Red} 2} & {\color{Red} 8} & {\color{Red} 7} & 4 & 1 & 9 & {\color{Red} 6} & {\color{Red} 3} & 5 \\
{\color{Red} 3} & {\color{Red} 4} & {\color{Red} 5} & {\color{Red} 2} & 8 & {\color{Red} 6} & {\color{Red} 1} & 7 & 9
\end{bmatrix}
$$

<b>Example:</b><br>
Input: board =<br>
[["5","3",".",".","7",".",".",".","."],<br>
["6",".",".","1","9","5",".",".","."],<br>
[".","9","8",".",".",".",".","6","."],<br>
["8",".",".",".","6",".",".",".","3"],<br>
["4",".",".","8",".","3",".",".","1"],<br>
["7",".",".",".","2",".",".",".","6"],<br>
[".","6",".",".",".",".","2","8","."],<br>
[".",".",".","4","1","9",".",".","5"],<br>
[".",".",".",".","8",".",".","7","9"]]<br>
Output:<br>
[["5","3","4","6","7","8","9","1","2"],<br>
["6","7","2","1","9","5","3","4","8"],<br>
["1","9","8","3","4","2","5","6","7"],<br>
["8","5","9","7","6","1","4","2","3"],<br>
["4","2","6","8","5","3","7","9","1"],<br>
["7","1","3","9","2","4","8","5","6"],<br>
["9","6","1","5","3","7","2","8","4"],<br>
["2","8","7","4","1","9","6","3","5"],<br>
["3","4","5","2","8","6","1","7","9"]]<br><br>

<b>Constraints:</b><br>
board.length == 9<br>
board[i].length == 9<br>
board[i][j] is a digit or '.'.<br>
It is guaranteed that the input board has only one solution.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        isSolver(board);
    }
    bool isSolver(vector<vector<char>> &board)
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] == '.')
                {
                    for (char c = '1'; c <= '9'; c++)
                    {
                        if (isFeasible(board, i, j, c))
                        {
                            board[i][j] = c;
                            if (isSolver(board)) { return true; }
                            else { board[i][j] = '.'; }
                        }
                    }
                    // No feasible number
                    return false;
                }
            }
        }
        return true;
    }
    bool isFeasible(vector<vector<char>> board, int row, int col, char c)
    {
        for(int i = 0; i < 9; i++)
        {
            if(board[row][i] == c) { return false; }
            if(board[i][col] == c) { return false; }
            if(board[3*(row/3)+i/3][3*(col/3)+i%3] == c) { return false; }
        }
        return true;
    }
};
{% endhighlight %}

## 0038. Count and Say
<p align="justify">
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:<br>
countAndSay(1) = "1"<br>
countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.<br>
To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.<br><br>

<b>Example:</b><br>
Input: n = 1<br>
Output: "1"<br>
Explanation: This is the base case.<br><br>

Input: n = 4<br>
Output: "1211"<br>
Explanation:<br>
countAndSay(1) = "1"<br>
countAndSay(2) = say "1" = one 1 = "11"<br>
countAndSay(3) = say "11" = two 1's = "21"<br>
countAndSay(4) = say "21" = one 2 + one 1 = "12" + "11" = "1211"<br><br>

<b>Constraints:</b><br>
1 <= n <= 30<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    string countAndSay(int n) {
        string res = "1";
        if (n == 1) { return res; }
        for (int i = 2; i <= n; i++)
        {
            string cur = "";
            res.push_back('0');
            int len = int(res.length()), count = 1;
            char ch = res[0];
            for (int i = 1; i < len; i++)
            {
                if (res[i] == res[i-1]) { count++; }
                else
                {
                    cur += to_string(count);
                    cur.push_back(ch);
                    count = 1;
                    ch = res[i];
                }
            }
            res = cur;
        }
        return res;
    }
};
{% endhighlight %}

## 0039. Combination Sum
<p align="justify">
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order. The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different. It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.<br><br>

<b>Example:</b><br>
Input: candidates = [2,3,6,7], target = 7<br>
Output: [[2,2,3],[7]]<br>
Explanation:<br>
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.<br>
7 is a candidate, and 7 = 7.<br>
These are the only two combinations.<br><br>

Input: candidates = [2,3,5], target = 8<br>
Output: [[2,2,2,2],[2,3,3],[3,5]]<br><br>

Input: candidates = [2], target = 1<br>
Output: []<br><br>

Input: candidates = [1], target = 1<br>
Output: [[1]]<br><br>

Input: candidates = [1], target = 2<br>
Output: [[1,1]]<br><br>

<b>Constraints:</b><br>
1 <= candidates.length <= 30<br>
1 <= candidates[i] <= 200<br>
All elements of candidates are distinct.<br>
1 <= target <= 500<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> arr;
        int n = int(candidates.size());
        if (n == 0) { return res; }
        DFS(res, arr, candidates, n, target, 0);
        return res;
    }
    void DFS(vector<vector<int>> &res, vector<int> &arr,
             vector<int> candidates, int n, int target,
             int idx)
    {
        if (target == 0)
        {
            res.push_back(arr);
            return;
        }
        for (int i = idx; i < n; i++)
        {
            if (target-candidates[i] >= 0)
            {
                arr.push_back(candidates[i]);
                DFS(res, arr, candidates, n, target-candidates[i], i);
                arr.resize(arr.size()-1);
            }
        }
    }
};
{% endhighlight %}

## 0040. Combination Sum II*
<p align="justify">
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target. Each number in candidates may only be used once in the combination. Note: The solution set must not contain duplicate combinations.<br><br>

<b>Example:</b><br>
Input: candidates = [10,1,2,7,6,1,5], target = 8<br>
Output: <br>
[<br>
[1,1,6],<br>
[1,2,5],<br>
[1,7],<br>
[2,6]<br>
]<br><br>

Input: candidates = [2,5,2,1,2], target = 5<br>
Output: <br>
[<br>
[1,2,2],<br>
[5]<br>
]<br><br>

<b>Constraints:</b><br>
1 <= candidates.length <= 100<br>
1 <= candidates[i] <= 50<br>
1 <= target <= 30<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> arr;
        int n = int(candidates.size());
        if (n == 0) { return res; }
        map<int, int> dict;
        map<int, int>::iterator iter;
        for (int num: candidates)
        {
            iter = dict.find(num);
            if (iter == dict.end()) { dict[num] = 1; }
            else { dict[num]++; }
        }
        vector<pair<int, int>> nums;
        int size = 0;
        for (iter = dict.begin(); iter != dict.end(); iter++)
        {
            nums.emplace_back(iter->first, iter->second);
            size++;
        }
        DFS(res, arr, candidates, target, 0, size, nums);
        return res;
    }
    void DFS(vector<vector<int>> &res, vector<int> &arr,
             vector<int> candidates, int target, int idx,
             int size, vector<pair<int, int>> nums)
    {
        if (target <= 0 || idx >= size)
        {
            if (target == 0) { res.push_back(arr); }
            return;
        }
        for (int j = 1; j <= nums[idx].second; j++)
        {
            for (int k = 0; k < j; k++) { arr.push_back(nums[idx].first); }
            DFS(res, arr, candidates, target-j*nums[idx].first, idx+1, size,
                nums);
            for (int k = 0; k < j; k++) { arr.pop_back(); }
        }
        DFS(res, arr, candidates, target, idx+1, size, nums);
    }
};
{% endhighlight %}

## 0041. First Missing Positive*
<p align="justify">
Given an unsorted integer array, find the smallest missing positive integer.<br><br>

<b>Example:</b><br>
Input: [1,2,0]<br>
Output: 3<br><br>

Input: [3,4,-1,1]<br>
Output: 2<br><br>

Input: [7,8,9,11,12]<br>
Output: 1<br><br>

<b>Follow up:</b><br>
Your algorithm should run in O(n) time and uses constant extra space.<br><br>

<b>Solution:</b>
$$
\begin{matrix}
3 & 4 & -1 & 1 \\
 & & \Downarrow & \\
3 & 4 & 5 & 1 \\
 & & \Downarrow & \\
3 & 4 & -5 & 1 \\
3 & 4 & -5 & -1 \\
-3 & 4 & -5 & -1
\end{matrix}
$$
</p>
{% highlight C++ %}
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = int(nums.size());
        for (int i = 0; i < n; i++)
        {
            if (nums[i] <= 0) { nums[i] = n + 1; }
        }
        for (int i = 0; i < n; i++)
        {
            int num = abs(nums[i]);
            if (num <= n)
            {
                nums[num-1] = -abs(nums[num-1]);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (nums[i] > 0) { return i + 1; }
        }
        return n + 1;
    }
};
{% endhighlight %}

## 0042. Trapping Rain Water*
{% highlight C++ %}
/*
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is
represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this
case, 6 units of rain water (blue section) are being
trapped.
*/
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0, n = int(height.size());
        if (n < 3) { return ans; }
        int i = 0, j = n-1;
        int leftMax = 0, rightMax = 0;
        while (i < j)
        {
            if (height[i] < height[j])
            {
                if (height[i] >= leftMax) { leftMax = height[i]; }
                else { ans += leftMax - height[i]; }
                i++;
            }
            else
            {
                if (height[j] >= rightMax) { rightMax = height[j]; }
                else { ans += rightMax - height[j]; }
                j--;
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0043. Multiply Strings
{% highlight C++ %}
/*
Input: num1 = "123", num2 = "456"
Output: "56088"
*/
class Solution {
public:
    string multiply(string num1, string num2) {
        string ans = "";
        if (num1 == "0" || num2 == "0") { return "0"; }
        int n2 = int(num2.length());
        for (int j = n2-1; j >= 0; j--)
        {
            string str = strMultiChar(num1, num2[j]);
            str.insert(str.length(), n2-1-j, '0');
            ans = strAdd(ans, str);
        }
        return ans;
    }
    string strAdd(string s1, string s2)
    {
        int n1 = int(s1.length()), n2 = int(s2.length());
        if (n1 < n2)
        {
            swap(s1, s2);
            swap(n1, n2);
        }
        string ans = "";
        int i = n1-1, j = n2-1, add1 = 0;
        while (i >= 0 && j >= 0)
        {
            int sum = (s1[i--] - '0') + (s2[j--] - '0') + add1;
            add1 = 0;
            if (sum > 9)
            {
                sum -= 10;
                add1 = 1;
            }
            char insertChar = '0' + sum;
            ans.insert(0, 1, insertChar);
        }
        while (i >= 0)
        {
            int sum = (s1[i--] - '0') + add1;
            add1 = 0;
            if (sum > 9)
            {
                sum -= 10;
                add1 = 1;
            }
            char insertChar = '0' + sum;
            ans.insert(0, 1, insertChar);
        }
        if (add1) { ans.insert(0, 1, '1'); }
        return ans;
    }
    string strMultiChar(string str, char c)
    {
        string ans = "";
        int n = int(str.length()), add1 = 0;
        for (int i = n-1; i >= 0; i--)
        {
            int res = (str[i] - '0') * (c - '0') + add1;
            add1 = 0;
            if (res > 9)
            {
                add1 = res / 10;
                res = res % 10;
            }
            char insertChar = '0' + res;
            ans.insert(0, 1, insertChar);
        }
        if (add1) { ans = to_string(add1) + ans; }
        return ans;
    }
};
{% endhighlight %}

## 0044. Wildcard Matching
{% highlight C++ %}
/*
Given an input string (s) and a pattern (p),
implement wildcard pattern matching with support
for '?' and '*' where:
'?' Matches any single character.
'*' Matches any sequence of characters (including
the empty sequence).
The matching should cover the entire input string
(not partial).
Input: s = "adceb", p = "*a*b"
Output: true
*/
class Solution {
public:
    bool isMatch(string s, string p) {
        int ns = int(s.length()), np = int(p.length());
        int **dp = new int *[ns+1];
        for (int i = 0; i <= ns; i++) { dp[i] = new int [np+1]{}; }
        dp[ns][np] = true;
        for (int i = ns; i >= 0; i--)
        {
            for (int j = np-1; j >= 0; j--)
            {
                bool ijMat = i < ns && (s[i] == p[j] || p[j] == '?' ||
                                        p[j] == '*');
                if (p[j] == '*')
                {
                    dp[i][j] = dp[i][j+1] || (ijMat && dp[i+1][j]);
                }
                else { dp[i][j] = ijMat && dp[i+1][j+1];
                }
            }
        }
        bool ans = dp[0][0];
        for (int i = 0; i <= ns; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0045. Jump Game II*
{% highlight C++ %}
/*
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach
the last index is 2. Jump 1 step from index 0 to 1,
then 3 steps to the last index.
*/
class Solution {
public:
    int jump(vector<int>& nums)
    {
        int n = int(nums.size());
        if (n <= 1) { return 0; }
        int ans = 0, end = 0, farestPos = 0;
        for (int i = 0; i < n-1; i++)
        {
            farestPos = max(i + nums[i], farestPos);
            if (i == end)
            {
                end = farestPos;
                ans++;
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0046. Permutations
<p align="justify">
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
</p>
{% highlight C++ %}
/*
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Input: nums = [0,1]
Output: [[0,1],[1,0]]

Input: nums = [1]
Output: [[1]]
*/
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        int n = int(nums.size());
        vector<vector<int>> ans;
        if (n == 0) { return ans; }
        permute(nums, 0, n, ans);
        return ans;
    }
    void permute(vector<int> &nums, int idx, int n,
                 vector<vector<int>> &ans)
    {
        if (idx == n)
        {
            ans.push_back(nums);
            return;
        }
        for (int i = idx; i < n; i++)
        {
            swap(nums[i], nums[idx]);
            permute(nums, idx+1, n, ans);
            swap(nums[i], nums[idx]);
        }
    }
};
{% endhighlight %}

## 0047. Permutations II*
<p align="justify">
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
</p>
{% highlight C++ %}
/*
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
*/
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ans;
        int n = (int)nums.size();
        if (n == 0) { return ans; }
        sort(nums.begin(), nums.end());
        bool *isVisited = new bool [n]{};
        vector<int> arr;
        permute(nums, 0, n, ans, arr, isVisited);
        delete []isVisited;
        return ans;
    }
    void permute(vector<int> &nums, int idx, int n,
                 vector<vector<int>> &ans,
                 vector<int> &arr,
                 bool *isVisited)
    {
        if (idx == n)
        {
            ans.push_back(arr);
            return;
        }
        for (int i = 0; i < n; i++)
        {
            if (isVisited[i] || (i > 0 && nums[i] == nums[i-1] &&
                                 !isVisited[i-1])) { continue; }
            isVisited[i] = true;
            arr.emplace_back(nums[i]);
            permute(nums, idx+1, n, ans, arr, isVisited);
            isVisited[i] = false;
            arr.pop_back();
        }
    }
};
{% endhighlight %}

## 0053. Maximum Subarray*
<p align="justify">
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum. Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
</p>
{% highlight C++ %}
/*
Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Input: nums = [1]
Output: 1

Input: nums = [0]
Output: 0

Input: nums = [-1]
Output: -1

Input: nums = [-2147483647]
Output: -2147483647

Constraints:
1 <= nums.length <= 2 * 10^4
-2^31 <= nums[i] <= 2^31 - 1
*/
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = int(nums.size());
        int maxSum = -(1ll << 31), curSum = 0;
        for (int i = 0; i < n; i++)
        {
            if (curSum >= 0) { curSum += nums[i]; }
            else { curSum = nums[i]; }
            if (curSum > maxSum) { maxSum = curSum; }
        }
        return maxSum;
    }
};
{% endhighlight %}

## 0055. Jump Game
{% highlight C++ %}
/*
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1,
then 3 steps to the last index.
*/
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = int(nums.size());
        if (n == 0) { return false; }
        int count = 0, end = 0, maxPos = 0;
        for (int i = 0; i < n - 1; i++)
        {
            maxPos = max(nums[i] + i, maxPos);
            if (i == end)
            {
                end = maxPos;
                count++;
            }
        }
        return end >= n - 1;
    }
};
{% endhighlight %}

## 0058. Length of Last Word
<p align="justify">
Given a string s consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return 0. A word is a maximal substring consisting of non-space characters only.
</p>
{% highlight C++ %}
/*
Input: s = "Hello World"
Output: 5
*/
class Solution {
public:
    int lengthOfLastWord(string s) {
        int ans = 0, n = int(s.length());
        if (n == 0) { return ans; }
        bool isCount = false;
        int i = n - 1;
        while (i >= 0)
        {
            if (s[i] == ' ' && isCount) { break; }
            if (s[i] == ' ')
            {
                i--;
                continue;
            }
            ans++;
            i--;
            isCount = true;
        }
        return ans;
    }
};
{% endhighlight %}

## 0061. Rotate List
{% highlight C++ %}
/*
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
*/
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (head == nullptr || k <= 0) { return head; }
        ListNode *p1 = head, *p2 = head;
        int num = 1;
        while (p2->next != nullptr)
        {
            num++;
            p2 = p2->next;
        }
        k = k % num;
        k = num - k;
        while (--k)
        {
            head = head->next;
        }
        p2->next = p1;
        p1 = head;
        head = head->next;
        p1->next = nullptr;
        return head;
    }
};
{% endhighlight %}

## 0062. Unique Paths*
<p align="justify">
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below). The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below). How many possible unique paths are there?
$$
\begin{aligned}
& dp[i][1] = 1, \quad i = 1, 2, ..., m \\
& dp[1][j] = 1, \quad j = 1, 2, ..., n \\
& dp[i][j] = dp[i-1][j] + dp[i][j-1], \quad i = 2, 3, ..., m \quad j = 2, 3, ..., n
\end{aligned}
$$
</p>
{% highlight C++ %}
/*
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the
bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
*/
class Solution {
public:
    int uniquePaths(int m, int n) {
        int **dp = new int *[m];
        for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
        for (int i = 0; i < m; i++) { dp[i][0] = 1; }
        for (int j = 0; j < n; j++) { dp[0][j] = 1; }
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        int ans = dp[m-1][n-1];
        for (int i = 0; i < m; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0063. Unique Paths II
<p align="justify">
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below). The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below). Now consider if some obstacles are added to the grids. How many unique paths would there be? An obstacle and space is marked as 1 and 0 respectively in the grid.
</p>
{% highlight C++ %}
/*
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
*/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        vector<vector<int>> mat = obstacleGrid;
        int m = int(mat.size());
        if (m == 0) { return 0; }
        int n = int(mat[0].size());
        if (n == 0) { return 0; }
        int **dp = new int *[m];
        for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
        if (mat[0][0]) { return 0; }
        dp[0][0] = 1;
        for (int i = 1; i < m; i++)
        {
            if (mat[i][0]) { dp[i][0] = 0; }
            else { dp[i][0] = dp[i-1][0]; }
        }
        for (int j = 1; j < n; j++)
        {
            if (mat[0][j]) { dp[0][j] = 0; }
            else { dp[0][j] = dp[0][j-1]; }
        }
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                if (mat[i][j]) { dp[i][j] = 0; }
                else { dp[i][j] = dp[i-1][j] + dp[i][j-1]; }
            }
        }
        int ans = dp[m-1][n-1];
        for (int i = 0; i < m; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0064. Minimum Path Sum
<p align="justify">
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path. Note: You can only move either down or right at any point in time.
</p>
{% highlight C++ %}
/*
1 3 1
1 5 1
4 2 1
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
*/
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = int(grid.size());
        if (m == 0) { return 0; }
        int n = int(grid[0].size());
        if (n == 0) { return 0; }
        int **dp = new int *[m];
        for (int i = 0; i < m; i++) { dp[i] = new int [n]{}; }
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) { dp[i][0] = dp[i-1][0] + grid[i][0]; }
        for (int j = 1; j < n; j++) { dp[0][j] = dp[0][j-1] + grid[0][j]; }
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        int ans = dp[m-1][n-1];
        for (int i = 0; i < m; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0066. Plus One
<p align="justify">
Given a non-empty array of decimal digits representing a non-negative integer, increment one to the integer. The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit. You may assume the integer does not contain any leading zero, except the number 0 itself.
</p>
{% highlight C++ %}
/*
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.

Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.

Input: digits = [0]
Output: [1]
*/
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        vector<int> ans;
        vector<int>::iterator iter;
        int n = int(digits.size()), add = 1;
        for (int i = n-1; i >= 0; i--)
        {
            int res = digits[i] + add;
            iter = ans.begin();
            if (res >= 10)
            {
                ans.insert(iter, res-10);
                add = 1;
            }
            else
            {
                ans.insert(iter, res);
                add = 0;
            }
        }
        iter = ans.begin();
        if (add) { ans.insert(iter, 1); }
        return ans;
    }
};
{% endhighlight %}

## 0067. Add Binary
<p align="justify">
Given two binary strings a and b, return their sum as a binary string.
</p>
{% highlight C++ %}
/*
Input: a = "11", b = "1"
Output: "100"

Input: a = "1010", b = "1011"
Output: "10101"
*/
class Solution {
public:
    string addBinary(string a, string b) {
        string str = "";
        int aLen = int(a.length()), bLen = int(b.length());
        int i = aLen - 1, j = bLen - 1, add = 0;
        if (aLen < bLen)
        {
            swap(a, b);
            swap(aLen, bLen);
            swap(i, j);
        }
        while (i >= 0 && j >= 0)
        {
            int sum = a[i] - '0' + b[j] - '0' + add;
            if (sum >= 2)
            {
                char tempChar = '0' + sum - 2;
                str.insert(0, 1, tempChar);
                add = 1;
            }
            else
            {
                char tempChar = '0' + sum;
                str.insert(0, 1, tempChar);
                add = 0;
            }
            i--;
            j--;
        }
        while (i >= 0)
        {
            int sum = a[i] - '0' + add;
            if (sum >= 2)
            {
                char tempChar = '0' + sum - 2;
                str.insert(0, 1, tempChar);
                add = 1;
            }
            else
            {
                char tempChar = '0' + sum;
                str.insert(0, 1, tempChar);
                add = 0;
            }
            i--;
        }
        if (add == 1) { str.insert(0, 1, '1'); }
        return str;
    }
};
{% endhighlight %}

## 0069 Sqrt(x)*
<p align="justify">
Given a non-negative integer x, compute and return the square root of x. Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.
</p>
{% highlight C++ %}
/*
Input: x = 4
Output: 2

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and
since the decimal part is truncated, 2 is returned.
*/
class Solution {
public:
    int mySqrt(int x)
    {
        // Binary search
        long lo = 1, hi = x;
        while (lo <= hi)
        {
            long mi = (lo + hi) / 2;
            if (mi * mi < x) { lo = mi + 1; }
            else if (mi * mi > x) { hi = mi - 1; }
            else { return int(mi); }
        }
        return int(hi);
    }
};
{% endhighlight %}

## 0070. Climbing Stairs
<p align="justify">
You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
</p>
{% highlight C++ %}
/*
Example:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

Constraints:
1 <= n <= 45
*/
class Solution {
public:
    int climbStairs(int n) {
        int a = 1, b = 2;
        if (n <= 2) { return n; }
        for (int i = 3; i <= n; i++)
        {
            int c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
};
{% endhighlight %}

## 0072. Edit Distance
<p align="justify">
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. You have the following three operations permitted on a word: Insert a character, Delete a character, Replace a character
</p>
{% highlight C++ %}
/*
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
*/
class Solution {
public:
    int minDistance(string word1, string word2) {
        string s1 = word1, s2 = word2;
        int n1 = int(s1.length()), n2 = int(s2.length());
        int **dp = new int *[n1+1];
        for (int i = 0; i <= n1; i++) { dp[i] = new int [n2+1]{}; }
        for (int i = 0; i <= n1; i++)
        {
            for (int j = 0; j <= n2; j++)
            {
                if (i == 0 || j == 0) { dp[i][j] = max(i, j); }
                else
                {
                    int a = dp[i-1][j] + 1;
                    int b = dp[i][j-1] + 1;
                    int c = dp[i-1][j-1] + (s1[i-1] != s2[j-1]);
                    dp[i][j] = min(min(a, b), c);
                }
            }
        }
        int ans = dp[n1][n2];
        for (int i = 0; i <= n1; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0076. Minimum Window Substring*
{% highlight C++ %}
/*
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
*/
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> tDict, sDict;
        for (const auto &c : t) { tDict[c]++; }
        int ns = int(s.length());
        int l = 0, r = -1, ansL = -1, minLen = INT_MAX;
        while (r < ns)
        {
            if (tDict.find(s[++r]) != tDict.end()) { sDict[s[r]]++; }
            while (isCover(tDict, sDict) && l <= r)
            {
                if (r - l + 1 < minLen)
                {
                    minLen = r - l + 1;
                    ansL = l;
                }
                if (tDict.find(s[l]) != tDict.end())
                {
                    sDict[s[l]]--;
                }
                l++;
            }
        }
        return -1 == ansL ? string() : s.substr(ansL, minLen);
    }
    bool isCover(unordered_map<char, int> &tDict,
                 unordered_map<char, int> &sDict)
    {
        for (const auto &item : tDict)
        {
            if (sDict[item.first] < item.second) { return false; }
        }
        return true;
    }
};
{% endhighlight %}

## 0080. Remove Duplicates from Sorted Array II*
<p align="justify">
Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length. Do not allocate extra space for another array; you must do this by modifying the input array in-place with O(1) extra memory.
</p>
{% highlight C++ %}
/*
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3]

Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3]
*/
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int n = int(nums.size());
        if (n < 3) { return n; }
        int len = 1, count = 1;
        for (int i = 1; i < n; i++)
        {
            if (nums[i] == nums[i-1]) { count++; }
            else { count = 1; }
            if (count <= 2) { nums[len++] = nums[i]; }
        }
        return len;
    }
};
{% endhighlight %}

## 0086. Partition List
<p align="justify">
Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x. You should preserve the original relative order of the nodes in each of the two partitions.
</p>
{% highlight C++ %}
/*
from 1 -> 4 -> 3 -> 2 -> 5 -> 2
to   1 -> 2 -> 2 -> 4 -> 3 -> 5
*/
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        if (head == nullptr) { return head; }
        ListNode *h1 = new ListNode(0), *h2 = new ListNode(0);
        ListNode *p1 = head, *p2 = head->next, *cur1 = h1, *cur2 = h2;
        while (p1 != nullptr)
        {
            if (p1->val < x)
            {
                cur1->next = p1;
                cur1 = p1;
            }
            else
            {
                cur2->next = p1;
                cur2 = p1;
            }
            p1 = p2;
            if (p2 == nullptr) { break; }
            p2 = p2->next;
        }
        cur2->next = nullptr;
        cur1->next = h2->next;
        head = h1->next;
        delete h1;
        delete h2;
        return head;
    }
};
{% endhighlight %}

## 0094. Binary Tree Inorder Traversal*
{% highlight C++ %}
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        TreeNode *curNode = root;
        while (curNode != nullptr)
        {
            while (curNode != nullptr)
            {
                if (curNode->right != nullptr) { st.push(curNode->right); }
                st.push(curNode);
                curNode = curNode->left;
            }
            curNode = st.top();
            st.pop();
            while (!st.empty() && curNode->right == nullptr)
            {
                res.push_back(curNode->val);
                curNode = st.top();
                st.pop();
            }
            res.push_back(curNode->val);
            if (!st.empty())
            {
                curNode = st.top();
                st.pop();
            }
            else { curNode = nullptr; }
        }
        return res;
    }
};
{% endhighlight %}

## 0096. Unique Binary Search Trees
<p align="justify">
Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.
$$
\begin{aligned}
& C_{n} =
\begin{cases}
1, &\quad n = 0 \\
\frac{2(2n+1)}{n+2} C_{n-1}, &\quad n = 1, 2, 3, ...
\end{cases} \\
& C_{n} = \frac{(2n)!}{(n+1)! n!}
\end{aligned}
$$
</p>
{% highlight C++ %}
/*
Input: n = 3
Output: 5

Input: n = 1
Output: 1
*/
class Solution {
public:
    int numTrees(int n) {
        long C = 1;
        for (int i = 0; i < n; i++)
        {
            C = C * 2 * (2 * i + 1) / (i + 2) ;
        }
        return int(C);
    }
};
{% endhighlight %}

## 0098. Validate Binary Search Tree
<p align="justify">
Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST is defined as follows:<br>
The left subtree of a node contains only nodes with keys less than the node's key.<br>
The right subtree of a node contains only nodes with keys greater than the node's key.<br>
Both the left and right subtrees must also be binary search trees.
</p>
{% highlight C++ %}
/*
   2
 /   \
1     3
is a BST

    5
  /   \
 1     4
      / \
     3   6
is not a BST
*/
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        TreeNode *prec = nullptr;
        return isBST(root, prec);
    }
    bool isBST(TreeNode* root, TreeNode *&prec)
    {
        if (root != nullptr)
        {
            bool left = isBST(root->left, prec);
            if (left == false) { return false; }
            if (prec == nullptr) { prec = root; }
            else if (prec->val >= root->val) { return false; }
            prec = root;
            bool right = isBST(root->right, prec);
            if (right == false) { return false; }
        }
        return true;
    }
};
{% endhighlight %}

## 0100. Same Tree
{% highlight C++ %}
/*
Input: p = [1,2,3], q = [1,2,3]
Output: true
*/
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) { return true; }
        if ((p == nullptr && q != nullptr) ||
            (p != nullptr && q == nullptr) ||
            (p->val != q->val)) { return false; }
        bool left = isSameTree(p->left, q->left);
        if (!left) { return false; }
        bool right = isSameTree(p->right, q->right);
        if (!right) { return false; }
        return true;
    }
};
{% endhighlight %}

## 0101. Symmetric Tree
{% highlight C++ %}
/*
    1
   / \
  2   2
 / \ / \
3  4 4  3
*/
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if (root == nullptr) { return true; }
        return isSymmetricTree(root->left, root->right);
    }
    bool isSymmetricTree(TreeNode *leftTree, TreeNode *rightTree)
    {
        if (leftTree == nullptr && rightTree == nullptr) { return true; }
        if ((leftTree != nullptr && rightTree == nullptr) ||
            (leftTree == nullptr && rightTree != nullptr) ||
            leftTree->val != rightTree->val) { return false; }
        bool left = isSymmetricTree(leftTree->left, rightTree->right);
        if (!left) { return false; }
        bool right = isSymmetricTree(leftTree->right, rightTree->left);
        if (!right) { return false; }
        return true;
    }
};
{% endhighlight %}

## 0102. Binary Tree Level Order Traversal
{% highlight C++ %}
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        vector<int> arr;
        if (root == nullptr) { return ans; }
        queue<TreeNode*> qTree;
        qTree.push(root);
        TreeNode *front = root, *last = root, *nextLast = root;
        while(!qTree.empty())
        {
            front = qTree.front();
            qTree.pop();
            arr.emplace_back(front->val);
            if (front->left != nullptr)
            {
                qTree.push(front->left);
                nextLast = front->left;
            }
            if (front->right != nullptr)
            {
                qTree.push(front->right);
                nextLast = front->right;
            }
            if (front == last)
            {
                ans.emplace_back(arr);
                arr.clear();
                last = nextLast;
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0103. Binary Tree Zigzag Level Order Traversal
{% highlight C++ %}
/*
    3
   / \
  9  20
    /  \
   15   7
[
  [3],
  [20,9],
  [15,7]
]
*/
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        vector<int> arr;
        if (root == nullptr) { return ans; }
        queue<TreeNode*> qTree;
        qTree.push(root);
        TreeNode *front = root, *last = root, *nextLast = root;
        int level = 1;
        while(!qTree.empty())
        {
            front = qTree.front();
            qTree.pop();
            arr.emplace_back(front->val);
            if (front->left != nullptr)
            {
                qTree.push(front->left);
                nextLast = front->left;
            }
            if (front->right != nullptr)
            {
                qTree.push(front->right);
                nextLast = front->right;
            }
            if (front == last)
            {
                if (level % 2 == 0)
                {
                    int n = int(arr.size());
                    for (int i = 0; i < n/2; i++)
                    {
                        swap(arr[i], arr[n-1-i]);
                    }
                }
                ans.emplace_back(arr);
                arr.clear();
                last = nextLast;
                level++;
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0104. Maximum Depth of Binary Tree
<p align="justify">
Given the root of a binary tree, return its maximum depth. A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
</p>
{% highlight C++ %}
/*
    3
  /   \
 9    20
     /  \
    15  7
max depth = 3
*/
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) { return 0; }
        int leftDep = maxDepth(root->left);
        int rightDep = maxDepth(root->right);
        return 1 + max(leftDep, rightDep);
    }
};
{% endhighlight %}

## 0110. Balanced Binary Tree
{% highlight C++ %}
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if (root == nullptr) { return true; }
        bool ans = true;
        getDepth(root, ans);
        return ans;
    }
    int getDepth(TreeNode *root, bool &isBal)
    {
        if (root == nullptr) { return 0; }
        int leftDep = getDepth(root->left, isBal);
        int rightDep = getDepth(root->right, isBal);
        if (abs(leftDep - rightDep) > 1) { isBal = false; }
        return 1 + max(leftDep, rightDep);
    }
};
{% endhighlight %}

## 0111. Minimum Depth of Binary Tree
<p align="justify">
Given a binary tree, find its minimum depth. The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node. Note: A leaf is a node with no children.
</p>
{% highlight C++ %}
/*
    3
  /   \
 9    20
     /  \
    15  7
min depth = 2
*/
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (root == nullptr) { return 0; }
        int leftDep = minDepth(root->left);
        int rightDep = minDepth(root->right);
        if (leftDep != 0 && rightDep != 0)
        {
            return 1 + min(leftDep, rightDep);
        }
        else { return 1 + max(leftDep, rightDep); }
    }
};
{% endhighlight %}

## 0112. Path Sum
<p align="justify">
Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum. A leaf is a node with no children.
</p>
{% highlight C++ %}
/*
                    5
                /       \
               4         8
              /        /   \
            11        13    4
           /  \            / \
          7    2          5   1
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true

     1
   /   \
  2     3
Input: root = [1,2,3], targetSum = 5
Output: false
*/
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) { return false; }
        if (root->left == nullptr && root->right == nullptr)
        {
            if (root->val == targetSum) { return true; }
            return false;
        }
        bool isInLeft = hasPathSum(root->left, targetSum-root->val);
        if (isInLeft) { return true; }
        bool isInRight = hasPathSum(root->right, targetSum-root->val);
        if (isInRight) { return true; }
        return false;
    }
};
{% endhighlight %}

## 0113. Path Sum II
<p align="justify">
Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's sum equals targetSum. A leaf is a node with no children.
</p>
{% highlight C++ %}
/*
                    5
                /       \
               4         8
              /        /   \
            11        13    4
           /  \            / \
          7    2          5   1
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
*/
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> ans;
        if (root == nullptr) { return ans; }
        vector<int> arr;
        findPaths(root, targetSum, ans, arr);
        return ans;
    }
    void findPaths(TreeNode *root, int resVal, vector<vector<int>> &ans,
                   vector<int> &arr)
    {
        if (root == nullptr) { return; }
        if (root->left == nullptr && root->right == nullptr)
        {
            if (resVal == root->val)
            {
                arr.emplace_back(root->val);
                ans.emplace_back(arr);
                arr.pop_back();
            }
            return;
        }
        arr.emplace_back(root->val);
        findPaths(root->left, resVal-root->val, ans, arr);
        findPaths(root->right, resVal-root->val, ans, arr);
        arr.pop_back();
    }
};
{% endhighlight %}

## 0120. Triangle
{% highlight C++ %}
/*
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom
is 2 + 3 + 5 + 1 = 11 (underlined above).
*/
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = int(triangle.size());
        if (n == 0) { return 0; }
        int **dp = new int *[n];
        for (int i = 0; i < n; i++) { dp[i] = new int [n]{}; }
        dp[0][0] = triangle[0][0];
        for (int i = 1; i < n; i++)
        {
            dp[i][0] = dp[i-1][0] + triangle[i][0];
            for (int j = 1; j < i; j++)
            {
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]) + triangle[i][j];
            }
            dp[i][i] = dp[i-1][i-1] + triangle[i][i];
        }
        int ans = *min_element(dp[n - 1], dp[n - 1] + n);
        for (int i = 0; i < n; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0121. Best Time to Buy and Sell Stock
<p align="justify">
Say you have an array for which the ith element is the price of a given stock on day i. If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit. Note that you cannot sell a stock before you buy one.<br><br>

<b>Example:</b><br>
Input: [7,1,5,3,6,4]<br>
Output: 5<br>
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5. Not 7-1 = 6, as selling price needs to be larger than buying price.<br><br>

Input: [7,6,4,3,1]<br>
Output: 0<br>
Explanation: In this case, no transaction is done, i.e. max profit = 0.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = int(prices.size()), maxPro = 0, minPri = (1ll << 31) - 1;
        for (int i = 0; i < n; i++)
        {
            minPri = minPri < prices[i] ? minPri : prices[i];
            maxPro = maxPro > prices[i] - minPri ? maxPro : prices[i] - minPri;
        }
        return maxPro;
    }
};
{% endhighlight %}

## 0144. Binary Tree Preorder Traversal
<p align="justify">
Given the root of a binary tree, return the preorder traversal of its nodes' values.<br><br>

<b>Example:</b><br>
Input: root = [1,null,2,3]<br>
Output: [1,2,3]<br><br>

Input: root = []<br>
Output: []<br><br>

Input: root = [1]<br>
Output: [1]<br><br>

Input: root = [1,2]<br>
Output: [1,2]<br><br>

Input: root = [1,null,2]<br>
Output: [1,2]<br><br>

<b>Constraints:</b><br>
The number of nodes in the tree is in the range [0, 100].<br>
-100 <= Node.val <= 100<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> ans;
        if (root == nullptr) { return ans; }
        TreeNode *curNode = root;
        stack<TreeNode*> st;
        st.push(root);
        while (!st.empty())
        {
            curNode = st.top();
            st.pop();
            ans.push_back(curNode->val);
            if (curNode->right != nullptr) { st.push(curNode->right); }
            if (curNode->left != nullptr) { st.push(curNode->left); }
        }
        return ans;
    }
};

class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> ans;
        if (root == nullptr) { return ans; }
        TreeNode *curNode = root;
        stack<TreeNode*> st;
        while (true)
        {
            while (curNode != nullptr)
            {
                ans.push_back(curNode->val);
                if (curNode->right != nullptr) { st.push(curNode->right); }
                curNode = curNode->left;
            }
            if (st.empty()) { break; }
            curNode = st.top();
            st.pop();
        }
        return ans;
    }
};
{% endhighlight %}

## 0174. Dungeon Game*
<p align="justify">
The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess. The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately. Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers). In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
</p>
{% highlight C++ %}
/*
-2(K)  -3    3
-5     -10   1
10     30   -5(P)
For example, given the dungeon below, the initial health of the
knight must be at least 7 if he follows the optimal path
RIGHT-> RIGHT -> DOWN -> DOWN.
*/
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int m = int(dungeon.size());
        if (m == 0) { return 1; }
        int n = int(dungeon[0].size());
        if (n == 0) { return 1; }
        vector<vector<int>> dp (m+1, vector<int>(n+1, INT_MAX));
        dp[m][n-1] = dp[m-1][n] = 1;
        for (int i = m-1; i >= 0; i--)
        {
            for (int j = n-1; j >= 0; j--)
            {
                int minVal = min(dp[i+1][j], dp[i][j+1]);
                dp[i][j] = max(minVal - dungeon[i][j], 1);
            }
        }
        int ans = dp[0][0];
        return ans;
    }
};
{% endhighlight %}

## 0209. Minimum Size Subarray Sum
<p align="justify">
Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
</p>
{% highlight C++ %}
/*
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
*/
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = int(nums.size());
        if (n == 0) { return 0; }
        int minLen = INT_MAX, l = 0, r = 0, sum = nums[0];
        while (r < n && l <= r)
        {
            if (sum < target)
            {
                if (r + 1 < n) { sum += nums[r+1]; }
                r++;
            }
            else
            {
                if (minLen > r - l + 1) { minLen = r - l + 1; }
                sum -= nums[l++];
            }
        }
        return minLen == INT_MAX ? 0 : minLen;
    }
};
{% endhighlight %}

## 0230. Kth Smallest Element in a BST
<p align="justify">
Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.
</p>
{% highlight C++ %}
/*
     3
   /   \
  1     4
   \
    2
k = 1, kth min = 1
*/
class Solution {
public:
    int kthSmallest(TreeNode* root, int &k) {
        if (root != nullptr)
        {
            int left = kthSmallest(root->left, k);
            if (left >= 0) { return left; }
            if (k == 1) { return root->val; }
            k--;
            int right = kthSmallest(root->right, k);
            if (right >= 0) { return right; }
        }
        return -1;
    }
};
{% endhighlight %}

## 0257. Binary Tree Paths
<p align="justify">
Given a binary tree, return all root-to-leaf paths. Note: A leaf is a node with no children.
</p>
{% highlight C++ %}
/*
Input:
   1
 /   \
2     3
 \
  5
Output: ["1->2->5", "1->3"]
Explanation: All root-to-leaf paths are: 1->2->5, 1->3
*/
class Solution {
public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<int> arr;
        vector<string> ans;
        if (root == nullptr) { return ans; }
        getPaths(root, ans, arr);
        return ans;
    }
    void getPaths(TreeNode* root, vector<string> &ans,
                  vector<int> &arr)
    {
        if (root == nullptr) { return; }
        if (root->left == nullptr && root->right == nullptr)
        {
            string str = "";
            for (int ele : arr) { str += to_string(ele) + "->"; }
            str += to_string(root->val);
            ans.emplace_back(str);
            return;
        }
        arr.emplace_back(root->val);
        getPaths(root->left, ans, arr);
        getPaths(root->right, ans, arr);
        arr.pop_back();
    }
};
{% endhighlight %}

## 279. Perfect Squares
{% highlight C++ %}
/*
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
*/
class Solution {
public:
    int numSquares(int n) {
        if (n < 1) { return 0; }
        int *dp = new int [n+1]{};
        dp[1] = 1;
        for (int i = 2; i <= n; i++)
        {
            if (i * i <= n) { dp[i * i] = 1; }
            if (dp[i]) { continue; }
            int min = INT_MAX;
            for (int j = 1; j <= i/2; j++)
            {
                if (min > dp[j] + dp[i-j])
                {
                    min = dp[j] + dp[i-j];
                }
            }
            dp[i] = min;
        }
        int ans = dp[n];
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0304. Range Sum Query 2D - Immutable*
{% highlight C++ %}
/*
Given matrix = [
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12
*/
class NumMatrix {
    vector<vector<int>> accum;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        int nRow = int(matrix.size());
        if (nRow == 0) { return; }
        int nCol = int(matrix[0].size());
        if (nCol == 0) { return; }
        accum.resize(nRow, vector<int> (nCol+1, 0));
        for (int i = 0; i < nRow; i++)
        {
            for (int j = 0; j < nCol; j++)
            {
                accum[i][j+1] = accum[i][j] + matrix[i][j];
            }
        }
    }
    int sumRegion(int row1, int col1, int row2, int col2) {
        int ans = 0;
        for (int i = row1; i <= row2; i++)
        {
            ans += accum[i][col2+1] - accum[i][col1];
        }
        return ans;
    }
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */
{% endhighlight %}

## 0343. Integer Break
{% highlight C++ %}
/*
Input: 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
*/
class Solution {
public:
    int integerBreak(int n) {
        if (n < 2) { return n; }
        if (n == 2 || n == 3) { return n-1; }
        int *dp = new int [n+1]{};
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        for (int i = 4; i <= n; i++)
        {
            int prod = 1;
            for (int j = 1; j <= i/2; j++)
            {
                if (prod < dp[j] * dp[i-j])
                {
                    prod = dp[j] * dp[i-j];
                }
            }
            dp[i] = prod;
        }
        int ans = dp[n];
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0344. Reverse String
{% highlight C++ %}
/*
Input: ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]
*/
class Solution {
public:
    void reverseString(vector<char>& s) {
        int n = int(s.size());
        if (n == 0) { return; }
        for (int i = 0; i < n/2; i++)
        {
            swap(s[i], s[n-1-i]);
        }
    }
};
{% endhighlight %}

## 0395. Longest Substring with At Least K Repeating Characters*
{% highlight C++ %}
/*
Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb",
as 'a' is repeated 2 times and 'b' is repeated
3 times.
*/
class Solution {
public:
    int longestSubstring(string s, int k) {
        int n = int(s.length()), ans = 0;
        for (int t = 1; t <= 26; t++)
        {
            int l = 0, r = 0, tot = 0, numLessK = 0;
            int *count = new int [26]{};
            while (r < n)
            {
                count[s[r] - 'a']++;
                if (count[s[r] - 'a'] == 1)
                {
                    tot++;
                    numLessK++;
                }
                if (count[s[r] - 'a'] == k)
                {
                    numLessK--;
                }
                while (tot > t)
                {
                    count[s[l] - 'a']--;
                    if (count[s[l] - 'a'] == k-1)
                    {
                        numLessK++;
                    }
                    if (count[s[l] - 'a'] == 0)
                    {
                        numLessK--;
                        tot--;
                    }
                    l++;
                }
                if (numLessK == 0)
                {
                    ans = max(ans, r - l + 1);
                }
                r++;
            }
            delete []count;
        }
        return ans;
    }
};
{% endhighlight %}

## 0516. Longest Palindromic Subsequence*
{% highlight C++ %}
/*
Given a string s, find the longest palindromic subsequence's
length in s. You may assume that the maximum length of s is
1000.
Input: "bbbab"
Output: 4
*/
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        int n = int(s.length());
        if (n <= 1) { return n; }
        int **dp = new int *[n];
        for (int i = 0; i < n; i++)
        {
            dp[i] = new int [n]{};
            dp[i][i] = 1;
        }
        for (int i = n-2; i >= 0; i--)
        {
            for (int j = i+1; j < n; j++)
            {
                if (s[i] == s[j]) { dp[i][j] = dp[i+1][j-1] + 2; }
                else { dp[i][j] = max(dp[i+1][j], dp[i][j-1]); }
            }
        }
        int ans = dp[0][n-1];
        for (int i = 0; i < n; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0543. Diameter of Binary Tree
<p align="justify">
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
</p>
{% highlight C++ %}
/*
Given a binary tree
          1
         / \
        2   3
       / \
      4   5
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
Note: The length of path between two nodes is represented by the
number of edges between them.
*/
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int max = 0;
        getDep(root, max);
        return max;
    }
    int getDep(TreeNode *root, int &maxLen)
    {
        if (root == nullptr) { return 0; }
        int leftDep = getDep(root->left, maxLen);
        int rightDep = getDep(root->right, maxLen);
        int curLen = leftDep + rightDep;
        maxLen = maxLen > curLen ? maxLen : curLen;
        return (leftDep > rightDep ? leftDep : rightDep) + 1;
    }
};
{% endhighlight %}

## 0647. Palindromic Substrings
{% highlight C++ %}
/*
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a",
"a", "aa", "aa", "aaa".
*/
class Solution {
public:
    int countSubstrings(string s) {
        int n = int(s.length()), ans = n;
        if (n == 0) { return ans; }
        bool **dp = new bool *[n];
        for (int i = 0; i < n; i++)
        {
            dp[i] = new bool [n]{};
            dp[i][i] = true;
        }
        for (int i = 1; i < n; i++)
        {
            if (s[i-1] == s[i])
            {
                dp[i-1][i] = true;
                ans++;
            }
        }
        for (int k = 2; k < n; k++)
        {
            for (int i = 0; i < n-k; i++)
            {
                int j = i + k;
                if (s[i] == s[j] && dp[i+1][j-1])
                {
                    dp[i][j] = true;
                    ans++;
                }
            }
        }
        for (int i = 0; i < n; i++) { delete []dp[i]; }
        delete []dp;
        return ans;
    }
};
{% endhighlight %}

## 0738. Monotone Increasing Digits*
<p align="justify">
Given a non-negative integer N, find the largest number that is less than or equal to N with monotone increasing digits. (Recall that an integer has monotone increasing digits if and only if each pair of adjacent digits x and y satisfy x <= y.)
</p>
{% highlight C++ %}
/*
Input: N = 10
Output: 9

Input: N = 1234
Output: 1234

Input: N = 332
Output: 299
*/
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        int ans = 0;
        string str = to_string(N);
        int len = int(str.length()), idx = -1;
        for (int i = 1; i < len; i++)
        {
            if (str[i] < str[i-1])
            {
                idx = i;
                break;
            }
        }
        if (idx == -1) { return N; }
        int pos = idx - 1;
        while (pos >= 1 && str[pos] == str[pos-1]) { pos--; }
        str[pos]--;  // current position decrease by 1
        for (int i = pos + 1; i < len; i++) { str[i] = '9'; }
        for (int i = len-1, pow = 1; i >= 0; i--, pow *= 10)
        {
            ans += (str[i] - '0') * pow;
        }
        return ans;
    }
};
{% endhighlight %}

## 0741. Cherry Pickup*
{% highlight C++ %}
/*
You are given an n x n grid representing a field of cherries,
each cell is one of three possible integers. 0 means the cell is
empty, so you can pass through, 1 means the cell contains a
cherry that you can pick up and pass through, or -1 means
the cell contains a thorn that blocks your way. Return the
maximum number of cherries you can collect by following
the rules below:
Starting at the position (0, 0) and reaching (n - 1, n - 1) by
moving right or down through valid path cells (cells with value 0 or 1).
After reaching (n - 1, n - 1), returning to (0, 0) by moving left
or up through valid path cells. When passing through a path cell
containing a cherry, you pick it up, and the cell becomes an empty cell 0.
If there is no valid path between (0, 0) and (n - 1, n - 1), then no
cherries can be collected.

Input: grid = [[0,1,-1],[1,0,-1],[1,1,1]]
Output: 5
Explanation: The player started at (0, 0) and went down, down,
right right to reach (2, 2). 4 cherries were picked up during
this single trip, and the matrix becomes [[0,1,-1],[0,0,-1],[0,0,0]].
Then, the player went left, up, up, left to return home, picking up
one more cherry. The total number of cherries picked up is 5, and
this is the maximum possible.

Input: grid = [[1,1,-1],[1,-1,1],[-1,1,1]]
Output: 0
*/
class Solution {
public:
    int cherryPickup(vector<vector<int>>& grid) {
        int n = int(grid.size());
        if (n == 0) { return 0; }
        vector<vector<vector<int>>> 
            dp(n, vector<vector<int>>(n, vector<int>(n, -1)));
        int ans = move(dp, grid, 0, 0, 0, n);
        return max(0, ans);
    }

    int move(vector<vector<vector<int>>> &dp,
             vector<vector<int>> &grid,
             int r1, int c1, int c2, int n)
    {
        int r2 = r1 + c1 - c2;
        if (r1 >= n || c1 >= n || r2 >= n || c2 >= n ||
            grid[r1][c1] == -1 ||
            grid[r2][c2] == -1) { return -2; }
        if (r1 == n-1 && c1 == n-1) { return grid[r1][c1]; }
        if (dp[r1][c1][c2] != -1) { return dp[r1][c1][c2]; }
        int ans = move(dp, grid, r1, c1+1, c2+1, n);
        ans = max(ans, move(dp, grid, r1+1, c1, c2+1, n));
        ans = max(ans, move(dp, grid, r1, c1+1, c2, n));
        ans = max(ans, move(dp, grid, r1+1, c1, c2, n));
        if (ans >= 0)
        {
            ans += grid[r1][c1] + (c1 != c2 || r1 != r2) *
                grid[r2][c2];
        }
        dp[r1][c1][c2] = ans;
        return ans;
    }
};
{% endhighlight %}

## 0766. Toeplitz Matrix
{% highlight C++ %}
/*
1 2 3 4
5 1 2 3
9 5 1 2
*/
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int m = int(matrix.size());
        if (m == 0) { return false; }
        int n = int(matrix[0].size());
        if (n == 0) { return false; }
        for (int j = 0; j < n; j++)
        {
            int first = matrix[0][j];
            for (int i = 0; i < min(m, n-j); i++)
            {
                if (first != matrix[i][i+j]) { return false; }
            }
        }
        for (int i = 1; i < m; i++)
        {
            int first = matrix[i][0];
            for (int j = 0; j < min(n, m-i); j++)
            {
                if (first != matrix[i+j][j]) { return false; }
            }
        }
        return true;
    }
};
{% endhighlight %}

## 0832. Flipping an Image
{% highlight C++ %}
/*
Input: [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]
*/
class Solution {
public:
    vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
        int m = int(A.size()), n = int(A[0].size());
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n/2; j++)
            {
                swap(A[i][j], A[i][n-1-j]);
            }
            for (int j = 0; j < n; j++)
            {
                A[i][j] = 1 - A[i][j];
            }
        }
        return A;
    }
};
{% endhighlight %}

## 0867. Transpose Matrix
{% highlight C++ %}
/*
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[1,4,7],[2,5,8],[3,6,9]]
*/
class Solution {
public:
    vector<vector<int>> transpose(vector<vector<int>>& matrix) {
        int m = int(matrix.size());
        if (m == 0) { return matrix; }
        int n = int(matrix[0].size());
        if (n == 0) { return matrix; }
        vector<vector<int>> ans(n, vector<int>(m, 0));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                ans[j][i] = matrix[i][j];
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 0896. Monotonic Array
{% highlight C++ %}
/*
Input: [1,2,2,3]
Output: true
*/
class Solution {
public:
    bool isMonotonic(vector<int>& A) {
        int n = int(A.size());
        if (n == 1) { return true; }
        bool isIncrease = false, isDecrease = false;
        for (int i = 1; i < n; i++)
        {
            if (A[i] > A[i-1]) { isIncrease = true; }
            if (A[i] < A[i-1]) { isDecrease = true; }
            if (isIncrease && isDecrease) { return false; }
        }
        return true;
    }
};
{% endhighlight %}

## 1052. Grumpy Bookstore Owner
{% highlight C++ %}
/*
Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], X = 3
Output: 16
Explanation: The bookstore owner keeps themselves not grumpy for the last 3
minutes. The maximum number of customers that can be satisfied = 1 + 1 + 1 +
1 + 7 + 5 = 16.
*/
class Solution {
public:
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
        int ans = 0, n = int(customers.size());
        if (n == 0) { return ans; }
        for (int i = 0; i < n; i++)
        {
            if (grumpy[i] == 0) { ans += customers[i]; }
        }
        int add = 0, maxAdd = 0;
        for (int i = 0; i < X; i++) { ans += customers[i] * grumpy[i]; }
        for (int i = 1; i < n - X + 1; i++)
        {
            add += -customers[i-1] * grumpy[i-1] +
            customers[i+X-1] * grumpy[i+X-1];
            if (maxAdd < add) { maxAdd = add; }
        }
        return ans + maxAdd;
    }
};
{% endhighlight %}

## 1143. Longest Common Subsequence
<p align="justify">
Given two strings text1 and text2, return the length of their longest common subsequence. A subsequence of a string is a new string generated from the original string with some characters(can be none) deleted without changing the relative order of the remaining characters. (eg, "ace" is a subsequence of "abcde" while "aec" is not). A common subsequence of two strings is a subsequence that is common to both strings. If there is no common subsequence, return 0.
</p>
{% highlight C++ %}
/*
Input: text1 = "abcde", text2 = "ace" 
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
*/
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        string s1 = text1, s2 = text2;
        int n1 = int(s1.length()), n2 = int(s2.length());
        int **dp = new int *[n1+1];
        for (int i = 0; i < n1+1; i++) { dp[i] = new int [n2+1]{}; }
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
        for (int i = 0; i < n1+1; i++) { delete []dp[i]; }
        delete []dp;
        return lcs;
    }
};
{% endhighlight %}

## 1178. Number of Valid Words for Each Puzzle*
<p align="justify">
With respect to a given puzzle string, a word is valid if both the following conditions are satisfied:<br>
word contains the first letter of puzzle.<br>
For each letter in word, that letter is in puzzle.<br>
For example, if the puzzle is "abcdefg", then valid words are "faced", "cabbage", and "baggage"; while invalid words are "beefed" (doesn't include "a") and "based" (includes "s" which isn't in the puzzle).<br>
Return an array answer, where answer[i] is the number of words in the given word list words that are valid with respect to the puzzle puzzles[i].
</p>
{% highlight C++ %}
/*
words = ["aaaa","asas","able","ability","actt","actor","access"], 
puzzles = ["aboveyz","abrodyz","abslute","absoryz","actresz","gaswxyz"]
Output: [1,1,3,2,4,0]
Explanation:
1 valid word for "aboveyz" : "aaaa" 
1 valid word for "abrodyz" : "aaaa"
3 valid words for "abslute" : "aaaa", "asas", "able"
2 valid words for "absoryz" : "aaaa", "asas"
4 valid words for "actresz" : "aaaa", "asas", "actt", "access"
There're no valid words for "gaswxyz" cause none of the words in
the list contains letter 'g'.
*/
class Solution {
public:
    vector<int> findNumOfValidWords(vector<string>& words,
                                    vector<string>& puzzles) {
        int n = int(puzzles.size());
        unordered_map<int, int> dict;
        for (const string &word : words)
        {
            int mask = 0;
            for (const char &c : word)
            {
                mask |= (1 << (c - 'a'));
            }
            if (__builtin_popcount(mask) <= 7)
            {
                dict[mask]++;
            }
        }
        vector<int> ans;
        for (const string &puzzle : puzzles)
        {
            int tot = 0;
            for (int choose = 0; choose < (1 << 6); choose++)
            {
                int mask = 0;
                for (int i = 0; i < 6; i++)
                {
                    if (choose & (1 << i))
                    {
                        mask |= (1 << (puzzle[i + 1] - 'a'));
                    }
                }
                mask |= (1 << (puzzle[0] - 'a'));
                tot += dict[mask];
            }
            ans.emplace_back(tot);
        }
        return ans;
    }
};
{% endhighlight %}

## 1585. Check If String Is Transformable With Substring Sort Operations*
<p align="justify">
Given two strings s and t, you want to transform string s into string t using the following operation any number of times:<br>
Choose a non-empty substring in s and sort it in-place so the characters are in ascending order.<br>
For example, applying the operation on the underlined substring in "14234" results in "12344".<br>
Return true if it is possible to transform string s into string t. Otherwise, return false.<br>
A substring is a contiguous sequence of characters within a string.<br><br>

<b>Example:</b><br>
Input: s = "84532", t = "34852"<br>
Output: true<br>
Explanation: You can transform s into t using the following sort operations:<br>
"84532" (from index 2 to 3) -> "84352"<br>
"84352" (from index 0 to 2) -> "34852"<br><br>

Input: s = "34521", t = "23415"<br>
Output: true<br>
Explanation: You can transform s into t using the following sort operations:<br>
"34521" -> "23451"<br>
"23451" -> "23415"<br><br>

Input: s = "12345", t = "12435"<br>
Output: false<br><br>

Input: s = "1", t = "2"<br>
Output: false<br><br>

<b>Constraints:</b><br>
s.length == t.length<br>
1 <= s.length <= 105<br>
s and t only contain digits from '0' to '9'.<br><br>

<b>Solution:</b>
</p>
{% highlight C++ %}
class Solution {
public:
    bool isTransformable(string s, string t) {
        vector<vector<int>> idx(10);
        vector<int> count(10);
        for (int i = 0; i < int(s.length()); i++)
        {
            idx[s[i]-'0'].push_back(i);
        }
        for (int i = 0; i < int(t.length()); i++)
        {
            int digit = t[i] - '0';
            if (count[digit] == idx[digit].size()) { return false; }
            for (int j = 0; j < digit; j++)
            {
                if (count[j] != idx[j].size() &&
                    idx[j][count[j]] < idx[digit][count[digit]])
                {
                    return false;
                }
            }
            count[digit]++;
        }
        return true;
    }
};
{% endhighlight %}

## 1748. Sum of Unique Elements
<p align="justify">
You are given an integer array nums. The unique elements of an array are the elements that appear exactly once in the array. Return the sum of all the unique elements of nums.
</p>
{% highlight C++ %}
/*
Input: nums = [1,2,3,2]
Output: 4
Explanation: The unique elements are [1,3], and the sum is 4.
*/
class Solution {
public:
    int sumOfUnique(vector<int>& nums) {
        int ans = 0;
        unordered_map<int, bool> dict;
        unordered_map<int, bool>::iterator iter;
        for (int ele : nums)
        {
            iter = dict.find(ele);
            if (iter == dict.end())
            {
                ans += ele;
                dict[ele] = true;
            }
            else if (dict[ele])
            {
                dict[ele] = false;
                ans -= ele;
            }
        }
        return ans;
    }
};
{% endhighlight %}

## 
<p align="justify">

</p>
{% highlight C++ %}

{% endhighlight %}