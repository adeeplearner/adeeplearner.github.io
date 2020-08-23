---
layout: post
mathjax: true
title: Hello world

# optional stuff
tags: [example]
# feature-img: "assets/img/banner_sunset3.png"
---

This is the first blog entry I have here. As I am beginning to get familiarised with this framework so this post serves as a practice area, as well as to keep all possible examples in one place.

### Lists
Bullets are done as:
* Bullet one
* Bullet two
* Bullet three

Numbered bullets are done as:
1. Number one
2. Number two
3. Number three

### Text + Emphasis
Written text in paragraph and other can be seperated into section. I am just writing this to have enough material for a paragraph to show how highlighted part of a blog works:
> It is important to highlight text sometimes, so it is visible like this

### Links
In markdown, links can be done as follows:
A link that opens in current opened tab: [Markdown](http://daringfireball.net/projects/markdown/syntax) 
A link that opens in a new tab: [Markdown](http://daringfireball.net/projects/markdown/syntax){:target="_blank"}

### Tables
Tables are quite easy to do, here is an example:

Content  | Type
------------- | -------------
hello  | word
world  | word
9      | integer
1.2    | float

### Images
Images can be embedded/included similar to how you include links:
![Geometric pattern with fading gradient]({{ site.baseurl }}/assets/img/banner_sunset.png)

Centre the image

{: style="text-align:center"}
![Geometric pattern with fading gradient]({{ site.baseurl }}/assets/img/banner_sunset.png)

Resize the image

![Geometric pattern with fading gradient]({{ site.baseurl }}/assets/img/banner_sunset.png){:height="360px" width="360px"}

### Code
Highlighting for code in Jekyll is done using Pygments or Rouge. This theme makes use of Rouge by default.

{% highlight js %}
// count to ten
for i in range(10+1):
    print(i)

// count to twenty
for j in range(20+1):
    print(j)
{% endhighlight %}

### Maths/Equations
This theme uses KaTeX to display maths. Equations such as $$S_n = a \times \frac{1-r^n}{1-r}$$ can be displayed inline.

Alternatively, they can be shown on a new line:

$$ f(x) = \int \frac{2x^2+4x+6}{x-2} $$

Argmin in latex:
$$ \underset{\mu_m, \sigma_m}{arg\,min} $$
