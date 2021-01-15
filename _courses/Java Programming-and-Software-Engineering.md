---
layout: page
title:  "Java Programming and Software Engineering"
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





## 1. Programming Foundations with JavaScript, HTML and CSS
## 1.3 JavaScript for Web Pages
<p align="justify">
<b>1.</b><br>
What is the purpose of initializing the global image variables to null in the green screen web page?<br>
A. The code can check if the image is null before processing it.<br>
B. Global variables must be initialized when they are declared.<br>
C. The code clears the canvas before performing the green screen algorithm.<br>
<b>Answer</b>: A.<br><br>

<b>2.</b><br>
You would like to display an alert message if the image variable for the foreground image fgImage is not loaded. Which two of the following expressions evaluate to true if the image is not ready?<br>
A. fgImage == null<br>
B. fgImage.complete()<br>
C. fgImage != null<br>
D. ! fgImage.complete()<br>
<b>Answer</b>: AD.<br><br>

<b>3.</b><br>
In which of the following code snippets does the program alert the user "x is null"?<br>
A.
</p>
{% highlight JavaScript %}
var x = null;
function myFunction() {
  var x = 2;
}
myFunction();
if (x == null) {
  alert("x is null");
}
{% endhighlight %}
<p align="justify">
B.
</p>
{% highlight JavaScript %}
var x = null;
function myFunction() {
  x = 2;
}
myFunction();
if (x == null) {
  alert("x is null");
}
{% endhighlight %}
<p align="justify">
C.
</p>
{% highlight JavaScript %}
var x = null;
function myFunction() {
  var x = 2;
  if (x == null) {
    alert("x is null");
  }
}
{% endhighlight %}
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>4.</b><br>
You have created the following file input element:
</p>
{% highlight JavaScript %}
<input type="file" onchange="loadImage()" >
{% endhighlight %}
<p align="justify">
Which of the following attributes can you add to restrict the file upload to image files?<br>
A. accept="image/*"<br>
B. id="finput"<br>
C. multiple="false"<br>
<b>Answer</b>: .<br><br>

<b>5.</b><br>
You have the following code excerpt to allow a user to select a file from the input element with ID "finput" and display it to a canvas with ID "can."
</p>
{% highlight JavaScript %}
var file = document.getElementById("finput");
var image = new SimpleImage(file);
var canvas = document.getElementById("can");
image.drawTo(canvas);
{% endhighlight %}
<p align="justify">
Which of the following do you need to add, so that this code will work in CodePen or on another web page?<br>
A. A script specifying where to find the JavaScript library for SimpleImage<br>
B. The value of the file input, such as:
</p>
{% highlight JavaScript %}
var img = file.value;
{% endhighlight %}
<p align="justify">
C. A context variable for the canvas, such as:
</p>
{% highlight JavaScript %}
var context = canvas.getContext(“2d”);
{% endhighlight %}
<p align="justify">
<b>Answer</b>: A.<br><br>

<b>6.</b><br>
You have two pixels to convert to grayscale, and you would like to determine visually whether your code is likely to be working, so you work an example by hand. The first pixel is orange and has rgb(255,153,51), and the second pixel is green and has rgb(51,153,51). Once the grayscale pixels are printed, which one should appear as a lighter gray (closer to white).<br>
A. First pixel<br>
B. Second pixel<br>
<b>Answer</b>: A.<br><br>

<b>7.</b><br>
You are building a web page, and you create a text input element and specify an element ID for it. Why did you do this?<br>
A. An ID is required in order to use an event handler.<br>
B. The id attribute is required for input elements.<br>
C. You would like to be able to reference the input element programmatically.<br>
<b>Answer</b>: C.<br><br>
</p>