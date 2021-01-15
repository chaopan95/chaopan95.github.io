---
layout: page
title:  "Data Science"
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


## 1. The Data Scientist’s Toolbox
### 1.1 Data Science Fundamentals
#### 1.1.1 What is Data Science?
<p align="justify">
Data science can involve statistics, computer science, mathematics, data cleaning and formatting, and data visualization. An Economist Special Report sums up this melange of skills well. There are 3 qualities: Volume, Velocity and Variety.<br>
<b>Volume</b>: more and more data is becoming increasingly available.<br>
<b>Velocity</b>: aata is being generated at an astonishing rate.<br>
<b>Variety</b>: The data we can analyse comes in many forms.<br><br>

Data science use data to answer questions. Data science is an intersection of tree fields: <b>hacking skills</b>, <b>math & statistics knowledge</b> and <b>substantive expertise</b>. 
<center><img src="https://raw.githubusercontent.com/chaopan1995/chaopan1995.github.io/master/_imgs/COURSES/DS/1_1_1_1.png"/></center>
</p>
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which of the following is an example of structured data?<br>
A. A database of individual's addresses, phone numbers, and post codes<br>
B. YouTube video transcript<br>
C. Satellite imagery of weather patterns<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which is NOT one of the three V's of Big Data?<br>
A. volume, vibrant, vital<br>
B. vast, versatile, vital<br>
C. visible, variety, vast<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
Which of these is NOT one of the main skills embodied by data scientists?<br>
A. Access to large data sets<br>
B. Hacking skills<br>
C. Substantive expertise<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.1.2 What is Data?
<p align="justify">
Data is a set of values of qualitative or quantitative variables.<br>
Set: in statistics, the population you are trying to discover something about<br>
Variables: measurement or characteristics of an item<br>
Qualitative variable: measurements or information about qualities<br>
Quatative variable: measurement or information about quantities or numerical items<br><br>

<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which of these is an example of a quantitative variable?<br>
A. occupation, clothing brand, birthplace<br>
B. weight, height, color<br>
C. age, latitude, gender<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
Quantitative variables are measured on ordered, continuous scales.<br>
A. True<br>
B. False<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
What is the most important thing in Data Science?<br>
A. Statistical inference<br>
B. Using the right software<br>
C. The question you are trying to answer<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.1.3 Getting Help
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which of these might be a good title for a forum post?<br>
A. Removing rows with NAs in data.frame using subset(), R 3.4.3<br>
B. URGENT! R isn't working!<br>
C. mean() doesn't do what it should<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which is a characteristic of a good question when posting to message boards?<br>
A. Begs for help without providing information<br>
B. Explicitly lists versions of software being used<br>
C. Is insulting or rude<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
Which is NOT a good strategy for finding the answer to a problem in this course?<br>
A. Explaining your problem to a friend/coworker<br>
B. Searching the course forum<br>
C. Googling the error message<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.1.4 The Data Science Process
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which of these is NOT an effective way to communicate the findings of your analysis?<br>
A. Write a blog post<br>
B. Save code locally on your computer<br>
C. Write a scientific article<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
What's the first step in the data science process?<br>
A. Generating the question<br>
B. Exploring the data<br>
C. Analyzing the data<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Why should you include links or citations to others' work?<br>
A. It helps other quickly find information you've reference.<br>
B. Others' work is more important than your.<br>
C. Citations prove how relevant your work is.<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
What does Hilary Parker suggest led to the popularity of the name "Dewey" in the late 1800s? (You may have to reference <a href="https://hilaryparker.com/2013/01/30/hilary-the-most-poisoned-baby-name-in-us-history/"> Hilary's blog post.</a> for this question.<br>
A. People named their daughters after Dewey, the famous opera singer.<br>
B. People named their daughters after Farrah Dewey, after Charlie's Angels.<br>
C. People named their daughters after George Dewey, after the Spanish-American War.<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.1.5 Module 1 Quiz
<p align="justify">
<b>1.</b><br>
Which of these is NOT one of the main skills embodied by data scientists?<br>
A. Hacking skills<br>
B. Machine learning<br>
C. Math and stats<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
What is the most important thing in Data Science?<br>
A. The question you are trying to answer<br>
B. Statistical inference<br>
C. Working with large data sets<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Which of these might be a good title for a forum post?<br>
A. URGENT! R isn't working!<br>
B. Removing rows with NAs in data.frame using subset(), R 3.4.3<br>
C. How do I get rnorm() to work?<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
What's the first step in the data science process?<br>
A. Communicate your findings<br>
B. Exploring the data<br>
C. Generating the question<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
Which of these is an example of a quantitative variable?<br>
A. Latitude<br>
B. Occupation<br>
C. Educational level<br>
<b>Answer:</b> A.<br><br>
</p>

### 1.2 R and RStudio
#### 1.2.1 Installing R
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
What does CRAN stand for?<br>
A. CRAN isn't an acronym<br>
B. Comprehensive R Archive Network<br>
C. Celebrating R's Array of Notebooks<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
What does base R focus on?<br>
A. Statistical analysis<br>
B. Artificial intelligence<br>
C. Image analysis<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
What is the output when you type into R: mean(mtcars$mpg) ?<br>
A. 20.09062<br>
B. 5.432<br>
C. mean()<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
Why are we using R for the course track?<br>
A. R is free.<br>
B. R is the only programming language data scientists use.<br>
C. R is the best cloud computing language.<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.2.2 Installing R Studio
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
What is RStudio?<br>
A. A graphical user interface for R<br>
B. A programming language<br>
C. An R package for machine learning<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which is NOT an option for a file type when you go to File > New File in RStudio?<br>
A. R Beamer Presentation<br>
B. R Markdown<br>
C. Shiny Web App<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.2.3 RStudio Tour
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
How do you see a command you have previously run and save it to source?<br>
A. History tab > Highlight command > To Source<br>
B. Environment tab > Open folder > Save<br>
C. File > History > Save<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What is the name of the quadrant in the bottom left corner of RStudio, in the default layout?<br>
A. History<br>
B. Plots<br>
C. Console<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
Which of the following is NOT one of the options available under the Global Options menu in Tools?<br>
A. Versions<br>
B. General<br>
C. Sweave<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
Using the Help menu, find out which of the following is one of the three species of Iris present in the base R dataset: iris.<br>
A. Acaulis<br>
B. Bufo<br>
C. Setosa<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.2.4 R Packages
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
How would you install the package ggplot2?<br>
A. install.packages("ggplot2")<br>
B. library("ggplot2")<br>
C. install.package("ggplot2")<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Using the help files, what is NOT a function included in the devtools package?<br>
A. aes()<br>
B. check()<br>
C. install_github()<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Which is NOT one of the main repositories?<br>
A. RDocumentation<br>
B. GitHub<br>
C. CRAN<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
What command lists your R version, operating system, and loaded packages?<br>
A. SESSIONINFO()<br>
B. sessionInfo()<br>
C. session.Info()<br>
<b>Answer:</b> B.<br><br>

<b>5.</b><br>
Install and load the KernSmooth R package. What does the copyright message say?<br>
A. Copyright M. P. Wand 1997-2009<br>
B. Copyright M. P. Wand 1990-2009<br>
C. Copyright Matthew Wand 1997-2009<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.2.5 Projects in R
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which is NOT a way to create a new Project?<br>
A. Session > New Project<br>
B. File > New Project<br>
C. Project toolbar > New Project<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What file extension do Projects in R use?<br>
A. .Rproj<br>
B. .pRoject<br>
C. .R<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Creating a new project from scratch will NOT do which of the following?<br>
A. Initiate version control<br>
B. Create a new folder<br>
C. Open a blank RStudio window<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.2.6 Module 2 Quiz
<p align="justify">
<b>1.</b><br>
What does base R focus on?<br>
A. Mapping<br>
B. Statistical analysis<br>
C. Artificial intelligence<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
What is RStudio?<br>
A. A graphical user interface for R<br>
B. Version control software<br>
C. A programming language<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
What is the name of the quadrant in the bottom left corner of RStudio, in the default layout?<br>
A. History<br>
B. Plots<br>
C. Console<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
What command lists your R version, operating system, and loaded packages?<br>
A. versions()<br>
B. Sessioninfo()<br>
C. sessionInfo()<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
What file extension do Projects in R use?<br>
A. .Rproj<br>
B. .R<br>
C. .RPROJECT<br>
<b>Answer:</b> A.<br><br>
</p>

### 1.3 Version Control and GitHub
#### 1.3.1 Version Control
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
I'm done editing a file, I need to ______ those changes then _______ them, and ______ it to the ________.<br>
A. Commit, merge, push, repository<br>
B. Pull, push, commit, branch<br>
C. Stage, commit, push, repository<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
What is a good example of a message to accompany a commit?<br>
A. Modified linear model of height to include new covariate, genotype<br>
B. Added genotype<br>
C. Updated thing<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Which of these is NOT true about using a version control system?<br>
A. Version control should only be used by experts<br>
B. Version control helps make sure that you do not lose work that you have done<br>
C. Version control minimizes the need to save different versions of the same file on your computer<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.3.2 GitHub and Git
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
On each repository page in GitHub, in the top right hand corner there are three options. They are:<br>
A. Commit, contributors, issues<br>
B. Watch, star, fork<br>
C. Pull, clone, fork<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
To make a new repository on GitHub, which can you NOT do?<br>
A. Profile > New repository<br>
B. Plus sign > New repository<br>
C. Profile > Repositories > New<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
What command can you use to change the name associated with each of your commits?<br>
A. git.config --global username "Jane Doe"<br>
B. git config --local username "Jane Doe"<br>
C. git config --global user.name "Jane Doe"<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
What command can you use to see your Git configuration?<br>
A. git load --list<br>
B. git config -settings<br>
C. git config --list<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
Which of the following will initiate a git repository locally?<br>
A. git init<br>
B. git remote add<br>
C. git boom<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.3.3 Linking Github and RStudio
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
In what quadrant of RStudio will you find the Git tab ?<br>
A. Environment<br>
B. Files<br>
C. Viewer<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What is the order of commands to send a file to GitHub from within RStudio?<br>
A. Commit > Stage > Push<br>
B. Commit > Push<br>
C. Stage > Commit message > Commit > Push<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
Which can you NOT do from within the Commit window of RStudio?<br>
A. See the differences between your original file and your updated file<br>
B. Stage files<br>
C. Pull and push content from the repository<br>
D. Write a commit message<br>
E. None of the above<br>
<b>Answer:</b> E.<br><br>
</p>

#### 1.3.4 Projects Under Version Control
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
What do you call it when you create a local copy of a repository that you will work on collaboratively with the original repository owner?<br>
A. Clone<br>
B. Branch<br>
C. Merge<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
What is the command to initialize git in a directory?<br>
A. git init<br>
B. cd ~/dir/name/of/path/to/file<br>
C. git add .<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
How do you add all of the contents of a directory to version control?<br>
A. git commit -m "Message"<br>
B. cd ~/dir/name/of/path/to/file<br>
C. git add .<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
How do you make a commit from within the command line?<br>
A. cd ~/dir/name/of/path/to/file<br>
B. git commit -m "Message"<br>
C. git init<br>
<b>Answer:</b> B.<br><br>
</p>

#### 1.3.5 Module 3 Quiz
<p align="justify">
<b>1.</b><br>
What is a good example of a message to accompany a commit?<br>
A. Modified linear model of height to include new covariate, genotype<br>
B. Fixed problem with linear model<br>
C. Updated thing<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
On each repository page in GitHub, in the top right hand corner there are three options. They are:<br>
A. Watch, star, fork<br>
B. Pull, clone, fork<br>
C. Commit, contributors, issues<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
Which of the following will initiate a git repository locally?<br>
A. git init<br>
B. git remote add<br>
C. git boom<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
What is the order of commands to send a file to GitHub from within RStudio?<br>
A. Commit > Push<br>
B. Stage > Commit message > Commit > Push<br>
C. Pull > Push > Commit<br>
<b>Answer:</b> B.<br><br>

<b>5.</b><br>
How do you add all of the contents of a directory to version control?<br>
A. git add .<br>
B. cd ~/dir/name/of/path/to/file<br>
C. git commit -m "Message"<br>
<b>Answer:</b> A.<br><br>
</p>

### 1.4 R Markdown, Scientific Thinking, and Big Data
#### 1.4.1 R Markdown
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
How would you strike through some text?<br>
A. \strikethrough\<br>
B. ~~strikethrough~~<br>
C. --strikethrough--<br>
<b>Answer:</b> B.<br><br>

<b>2.</b><br>
What is the format for including a link that appears as blue text in your markdown document?<br>
A. ![link.com](text that is shown)<br>
B. (text that is shown)[link.com]<br>
C. [text that is shown](link.com)<br>
<b>Answer:</b> C.<br><br>

<b>3.</b><br>
How do you produce bold text?<br>
A. **bold**<br>
B. _bold_<br>
C. ~~bold~~<br>
<b>Answer:</b> A.<br><br>

<b>4.</b><br>
How do you produce italicized text?<br>
A. __some text__<br>
B. **some text**<br>
C. *some text*<br>
<b>Answer:</b> C.<br><br>

<b>5.</b><br>
How do you produce your final document?<br>
A. Knit<br>
B. Crochet<br>
C. Macrame<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.4.2 Types of Data Science Questions
<p align="justify">
Types of data analysis<br>
$\bigstar$ 1) descriptive<br>
-- describe or summarize a set of data<br>
-- early analysis when receiving new data<br>
-- generate simple summaries about the samples and their measurements<br>
-- not for generalizing the results of the analysis to a larger population or trying to make conclusions<br>
$\bigstar$ 2) exploratory<br>
-- examine the data and find relationships that weren't previsouly known<br>
-- correlation does not imply causation<br>
$\bigstar$ 3) inferential<br>
-- use a relatively small smaple of data to say something about the population at large<br>
$\bigstar$ 4) predictive<br>
-- use current and historical data to make predictions about future data<br>
$\bigstar$ 5) causal<br>
-- see what happens to one variable when we manipulate another variable<br>
$\bigstar$ 6) mechanistic<br>
-- understand the exact changes in variables that lead to exact changes in other variables<br><br>

<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which of the following describes a predictive analysis?<br>
A. Finding if one variable is related to another one<br>
B. Showing the effect on a variable of changing the values of another variable<br>
C. Using data collected in the past to predict values in the future<br>
<b>Answer:</b> C.<br><br>

<b>2.</b><br>
We collect data on all the songs in the Spotify catalogue and want to summarize how many are country western, hip-hop, classic rock, or other. What type of analysis is this?<br>
A. Exploratory<br>
B. Descriptive<br>
C. Predictive<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
We collect data on a small sample of songs from the Spotify catalogue and want to figure out the relationship between the use of the word "truck" and whether a song is country western. What type of analysis is this?<br>
A. Descriptive<br>
B. Inferential<br>
C. Exploratory<br>
<b>Answer:</b> B.<br><br>
</p>

#### 1.4.3 Experimental Design
<p align="justify">
$\bigstar$ 1) formulate your question (in advance of any data collection)<br>
$\bigstar$ 2) design your experiment<br>
$\bigstar$ 3) indentify problems and sources of error<br>
$\bigstar$ 4) collect the data<br><br>

<b>Practice Quiz</b>:<br>
<b>1.</b><br>
In a study measuring the effect of diet on BMI, cholesterol, lipid levels, triglyceride levels, and glycemic index, which is an independent variable?<br>
A. Diet<br>
B. BMI<br>
C. Lipid levels<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which of the following is NOT a method to control your experiments?<br>
A. Control group<br>
B. Placebo effect<br>
C. Blinding<br>
<b>Answer:</b> B.<br><br>

<b>3.</b><br>
What might a confounder be in an experiment looking at the relationship between the prevalence of white hair in a population and wrinkles?<br>
A. Socioeconomic status<br>
B. Smoking status<br>
C. Age<br>
<b>Answer:</b> C.<br><br>

<b>4.</b><br>
According to Leek group recommendations, what data do you need to share with a collaborating statistician?<br>
A. The raw data<br>
B. A tidy data set<br>
C. A code book describing each variable and its values in the tidy data set<br>
D. An explicit and exact recipe of how you went from the raw data to the tidy data and the code book<br>
E. All of the above<br>
<b>Answer:</b> E.<br><br>

<b>5.</b><br>
If you set your signifance level at p-value ≤ 0.01, how many significant tests would you expect to see by chance if you carry out 1000 tests?<br>
A. 10<br>
B. 50<br>
C. 100<br>
<b>Answer:</b> A.<br><br>

<b>6.</b><br>
What is an experimental design tool that can be used to address variables that may be confounders at the design phase of an experiment?<br>
A. Stratifying variables<br>
B. Data cleaning<br>
C. Using all the data you have access to<br>
<b>Answer:</b> A.<br><br>

<b>7.</b><br>
Which of the following describes a descriptive analysis?<br>
A. Use your sample data distribution to make predictions for the future<br>
B. Draw conclusions from your sample data distribution and infer for the larger population<br>
C. Generate a table summarizing the number of observations in your dataset as well as the central tendencies and variances of each variable<br>
<b>Answer:</b> C.<br><br>
</p>

#### 1.4.4 Big Data
<p align="justify">
<b>Practice Quiz</b>:<br>
<b>1.</b><br>
Which is NOT one of the three V's of Big Data?<br>
A. Vexing<br>
B. Velocity<br>
C. Variety<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which one of the following is an example of structured data?<br>
A. A table of names and student grades<br>
B. Lung x-ray images<br>
C. The text from a series of books<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
What is the reason behind the explosion of interest in big data?<br>
A. The price and difficulty of collecting and storing data has dramatically dropped<br>
B. There have been massive improvements in machine learning algorithms<br>
C. There have been massive improvements in statistical analysis techniques<br>
<b>Answer:</b> A.<br><br>
</p>

#### 1.4.5 Module 4 Quiz
<p align="justify">
<b>1.</b><br>
What is the format for including a link that appears as blue text in your markdown document?<br>
A. [text that is shown](link.com)<br>
B. (link.com)[text that is shown]<br>
C. (text that is shown)[link.com]<br>
<b>Answer:</b> A.<br><br>

<b>2.</b><br>
Which of the following describes a predictive analysis?<br>
A. Using data collected in the past to predict values in the future<br>
B. Finding if one variable is related to another one<br>
C. Showing the effect on a variable of changing the values of another variable<br>
<b>Answer:</b> A.<br><br>

<b>3.</b><br>
We collect data on all the songs in the Spotify catalogue and want to summarize how many are country western, hip-hop, classic rock, or other. What type of analysis is this?<br>
A. Exploratory<br>
B. Descriptive<br>
C. Predictive<br>
<b>Answer:</b> B.<br><br>

<b>4.</b><br>
What might a confounder be in an experiment looking at the relationship between the prevalence of white hair in a population and wrinkles?<br>
A. Age<br>
B. Socioeconomic status<br>
C. Sex<br>
<b>Answer:</b> A.<br><br>

<b>5.</b><br>
Which one of the following is an example of structured data?<br>
A. The text from a series of books<br>
B. Lung x-ray images<br>
C. A table of names and student grades<br>
<b>Answer:</b> C.<br><br>
</p>


## 2. R Programming
### 2.1 Week 1
#### 2.1.1 Background, Getting Started, and Nuts & Bolts
<p align="justify">
<b>R Objects and Attributes</b><br>
$\bigstar$ Objects<br>
-- R has 5 basic or "atomic" classes of objects:<br> character, (real) number, integer, complex, logical (true and false)<br>
-- The most basic object is a vector: a vector can only contain objects of the same class; ist is represented as a vector but it can contain objects of different classes<br>
-- Empty vector can be reated with vector()<br>
$\bigstar$ Numbers<br>
-- numbers in R a generally treated as numeric objects (double precision real number)<br>
-- if you explicitly want an integer, you need to specify the L suffix, e.g. 1L explicitly give an integer while 1 gives a numeric object<br>
-- Inf means infinity, e.g. 1/0<br>
-- NaN represents an undefined value or a missing value<br>
$\bigstar$ Attributes<br>
-- R objects have attributes: names, dimnames, dimensions, class, length, other user-defined attributes<br>
-- attributes of an object can be accessed using attributes()<br><br>

<b>Vectors and Lists</b><br>
$\bigstar$ Mixing objects<br>
-- c(1.7, 'a', TRUE)<br>
$\bigstar$ Explicit coercion<br>
x <- 0:6<br>
as.character(x)<br>
'0' '1' '2' '3' '4' '5' '6'<br><br>

<b>Reading Tabular Data</b><br>
$\bigstar$ read.table<br>
data <- read.table('data.csv')<br>

</p>

#### 2.1.2 Quiz
<p align="justify">
<b>.</b><br>
<b>Answer:</b> .<br><br>

<b>.</b><br>
<b>Answer:</b> .<br><br>

<b>.</b><br>
<b>Answer:</b> .<br><br>

<b>.</b><br>
<b>Answer:</b> .<br><br>

<b>.</b><br>
<b>Answer:</b> .<br><br>

<b>.</b><br>
<b>Answer:</b> .<br><br>
</p>

