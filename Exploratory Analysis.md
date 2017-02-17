Exploratory Analysis with Python & R
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958), [Byte Academy](byteacademy.co) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python & Pip](#01-python--pip)
	+ [0.2 R & R Studio](#02-r--r-studio)
	+ [0.3 Other](#03-other)
- [1.0 Introduction](#10-introduction)
- [2.0 Data Normalization](#20-data-normalization)
- [3.0 Strings](#30-strings)
- [4.0 Missing Values](#40-missing-values)
- [5.0 Outlier Detection](#50-outlier-detection)
- [6.0 Final Words](#60-final-words)
	+ [6.2 Mini Courses](#62-mini-courses)


## 0.0 Setup

This guide was written in Python 3.5 and R 3.2.3.

### 0.1 Python & Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

Let's install the modules we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules: 

```
pip3 install 
```

### 0.2 R & R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("mice")
install.packages("VIM")
```

### 0.3 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows: 
```
pip3 install virtualenv
virtualenv your_env
```
And then launch it with: 
```
source your_env/bin/activate
```

Cool, now we're ready to start! 


## 1.0 Introduction

### 1.1 What is Exploratory Data Analysis?

Exploratory Data Analysis (EDA) is an approach for data analysis that employs a variety of techniques to do the following tasks:

- Maximize insight into a dataset
- Uncover the underlying structure of a dataset
- Detect outliers and anomalies
- Test underlying assumptions

### 1.2 EDA Techniques

In [this](learn.adicu.com/intro-dv) tutorial, what I didn't mention while overviewing the different types of graphs, is what they're often used for: Exploratory Data Analysis.

Classical techniques are usually quantitative and include things like t-tests, f-tests, and chi-squared tests. In Exploratory Data Analysis, the techniques are usually graphical, including scatter plots, character plots, box plots, histographs, probability plots, residual plots, and mean plots.



## 2.0 Exploratory Computing


### 2.1 Finding the Zero

Finding the zero of a function is a common task in exploratory computing. The value where the function equals zero is also called the root and finding the zero is referred to as root finding. There exists a number of methods to find the zero of a function varying from robust but slow (so it always finds a zero but it takes quite a few function evaluations) to fast but not so robust (it can find the zero very fast, but it won't always find it). Here we'll use the latter one:

Consider the function f(x) = 0.5−e<sup>−x</sup>. Let's find the root for this. First, we need to write a Python function for `f(x)`.

``` python
def f(x):
    return(0.5 - np.exp(-x))
```

We will use the method fsolve to find the zero of a function. `fsolve` is part of the `scipy.optimize` package. fsolve takes two arguments: the function and a starting value for the search.

``` python
from scipy.optimize import fsolve
xzero = fsolve(f,1)
print('result of fsolve:', xzero)
```

Which gets us: 
``` bash
result of fsolve: [ 0.69314718]
```

Now, we know that the actual root is equal to `x = -ln(0.5)`. So let's calculate that and compare the two values:

``` python
print('f(x) at xzero:   ', f(xzero))
print('exact value of xzero:', -np.log(0.5))
```

Which gets us:
``` bash
exact value of xzero: 0.69314718056
```

Notice that the two results are incredibly close!

### 2.2 Cumulative Density Function

Recall that the Cumulative Density Function provides us with the probability that x takes on a value less than x. 

The Cumulative Density Distribution F(x) of the Normal distribution is given by:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/normal%20cdf.png?raw=true "Logo Title Text 1")

Now, using the scipy module, we can create the CDF for a Normal Distribution:

``` python
from scipy.special import erf
```
where &mu; is the mean, &sigma; is the standard deviation, and `erf` is the error function. 


### 2.3 Continuous Random Variables

The most common probability distribution is the Normal distribution. Random numbers from a Normal distribution may be generated with the `standard_normal` function in the random subpackage of numpy. 

The numbers are drawn from a "standard" Normal distribution, which means a Normal distribution with mean 0 and standard deviation 1. The mean and standard deviation of a dataset can be computed with the functions `mean` and `std` of the numpy package.

So let's begin by importing the needed libraries: 

``` python
```

This line of code gets us 100 random numbers in an array:
``` python
data = rnd.standard_normal(100)
```

Now let's check out the mean and standard deviation:
``` python
np.mean(data)
np.std(data)
```

Note that the results aren't exactly 0 or 1 because they're only estimates of the true underlying mean and standard deviation.

Now, let's try a modified example:
``` python
mu = 6.0
sig = 2.0
data = sig * rnd.standard_normal(100) + mu
```

Next, let's see how a histogram emulates the normal distribution:
``` python
from scipy.stats import norm
a = plt.hist(data, bins=12, range=(0, 12), normed=True)
x = np.linspace(0, 12, 100)
y = norm.pdf(x, 6, 2) 
plt.plot(x, y, 'r')
plt.xlabel('bins')
plt.ylabel('probability');
```

### 2.4 Box Whisker

Box-whisker plots are a method to visualize the level and spread of the data. From a boxplot, you can see whether the data is symmetric or not, and how widely the data are spread. A box-whisker plot may be created with the boxplot function in the matplotlib package as follows

``` python
rnd.seed(10)
data = 2 * rnd.standard_normal(500) + 10.0 
a = plt.boxplot(data)
```

The blue box spans the IQR ranging from the lower quartile (25%) to the upper quartile (75%). The whiskers are the black lines that are connected to the 50% box with the blue dashed lines.



## 5.0 Final Words

### 5.1 Resources

