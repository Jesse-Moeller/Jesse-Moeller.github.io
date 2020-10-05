---
layout: post
title: "Linear Regression and Major League Baseball"
author: "Jesse Moeller"
categories: journal
tags: []
image: pablo_sandoval.jpg
---

I love the internet. There is no shortage of free learning material on the internet. I learned calculus on the internet. Any meaningful progress I have made towards learning a computer skill has happened on the internet. Naturally, the internet would be there to help me get into machine learning. There are several dozen starting points available to get into machine learning from ground zero. After reviewing several books, and at the suggestion of one of the mentors from my summer internship, Mike Grosskopf, I decided to crack open Elements of Statistical Learning (ESL), by Friedman, Hastie, and Tibshirani. The book is freely availabe [here](https://web.stanford.edu/~hastie/ElemStatLearn/). I really appreciate the way this book is layed out: there's a fundamental block of chapters which are, more or less, required by the others, but the rest of the book is a buffet of content and can be consumed in any order. Mike encouraged me to read through Chapter 3, one of those fundamental chapters, to try and learn about shrinkage methods.

Chapter 3 of ESL is about linear regression. When we want to predict a value $y$, like someone's weight, based on certain features $x$, like how old that person is, then a simpe way to do it is to just use a linear equation: $\hat{y}=mx+b$. So our model is entirely determined by these numbers $m$ and $b$. I guess, in this context, $b$ might mean "the average weight of a newborn". But how *good* is this model? How do we judge how well our $m$ and $b$ perform? A straightforward way to judge this is just to add up the errors of our predictions. To avoid the positive error and negative error cancelling eachother out, we will add up the squared error. Relative to the data, $m$ and $b$ are variables and so this process defines a function, $L(m,b)$, which we call the loss function. Based on the description above we have 

$$L(m,b) = \sum (\hat{y}-y)^2$$ 

If you know some calculus, then you can just minimize this equation to find the best $m$ and $b$ for the job. If we are predicting a value using more factors than just one, not much changes. The $x$ and $m$ and $b$ become vectors and that's it. Still, we can ask: how *good* is this model? Sure, objectively it is the minimum of this loss function, but there are other concerns. What if we have several predictors and we don't believe that each of these will contribute to the prediction? Then it is possible that we can get better prediction accuracy by shrinking some of the coefficients in $m$, or setting them to zero entirely. One way to penalize $m$ for having too many non-zero coefficients is to modify our loss function. Perhaps something like

$$
L(m,b) = \underbrace{\sum (\hat{y}-y)^2}_{\text{penalty for being far away from data}} + \underbrace{\lambda ||m||^2}_{\text{penalty for too many coefficients}}
$$

Like in the previous situation, since this function is nice, you can do some calculus to find a minimum. However, if we leave the square off of the new term in our loss function, then there is no closed form expression for the minimum. We can still find solutions, however, and in theory both of these models should reduce the number of coefficients that we see in $m$. The authors of ESL discuss why you might want to leave the square off, and the reasoning is geometric in nature but I will leave that out here. The three models I have mentioned so far are called linear regression, ridge regression, and lasso regression, respectively. There's another linear model which I won't explain in detail called elastic net and you can kind of think of it as somewhere between ridge and lasso. 

With this grounding, I wanted to see how shrinkage could be applied. I downloaded a Major League Baseball dataset and I set out to predict a player's weight from the other features in the dataset using a linear model. The features at my disposal were the player's team, their position, and their height, weight, and age. With each of these models I wanted to train the model on some of the data set (training data) and see how well it behaved on the remaining portion of data that it hasn't seen (the test data). This is standard practice during model selection, and the scikit-learn packages make this part very easy for us. Here's what that code looks like:

```python
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
import matplotlib as plt

# formatting the output images
plt.rcParams["figure.figsize"] = (8, 6)

# import MLB player weight data
names = ['Name','Team','Position','Height','Weight','Age']
MLB = read_csv(r"C:\Users\moell\Desktop\Machine Learning\Data\MLBweight.csv", names=names)
MLB=MLB.drop('Name', axis=1)

# one-hot encoding
MLB=pd.get_dummies(MLB)

# formatting X and y
y=np.array(MLB['Weight'])
MLB=MLB.drop('Weight', axis=1)
feature_list=list(MLB.columns)
X=np.array(MLB)
y=np.nan_to_num(y)

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# make a list of labelled models (name,model)
models=[('Linear Regression', LinearRegression()),
        ('Ridge', RidgeCV()),
        ('Lasso', LassoCV(cv=5, random_state=0)),
        ('Elastic Net', ElasticNetCV(cv=5, random_state=0))]

# compare Results
for (name,model) in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    print('Mean Absolute Error for', name, ' :', round(np.mean(errors), 2))

    # plotting the coefficients of the model
    coefs = pd.DataFrame({'Feature': feature_list,
                      'Importance': model.coef_})
    coefs = coefs.set_index('Feature').sort_values('Importance', ascending=False)
    coefs_plot = coefs.plot(kind='bar', title= name+' feature importance')
```

And here's the output:
```
Mean Absolute Error for Linear Regression  : 12.64
Mean Absolute Error for Ridge  : 12.55
Mean Absolute Error for Lasso  : 12.53
Mean Absolute Error for Elastic Net  : 12.58
```

Also, here are the "feature importance" plots

![image](/assets/img/mlb_weight_lr.png)
![image](/assets/img/mlb_weight_ridge.png)
![image](/assets/img/mlb_weight_lasso.png)
![image](/assets/img/mlb_weight_enet.png)

So, here's what sticks out to me. None of these models are significantly better at predicting the weight, at least as far as I can tell. However, looking at the plots of the coefficients, we see a huge difference between the basic linear regression and the other models. It is very clear that, in the other models, shrinkage is happening. We see a continuous shrinkage to zero with ridge, as expected. We see a sharp chop-off to zero with lasso, as expected (if you read ESL chapter 3). Lastly, we see something between these with elastic net, as expected (since it is kind of a mix of the previous two). So, hey! That's neat. Looking at the features themselves, we see that the shrinkage models don't exactly agree on everything but we get a sense that being a designated hitter or a catcher means you're heavier, and being a second baseman or a shortstop means you're lighter. These results also make sense, but I did not expect or predict them myself. Baseball isn't on the tips of my fingers in that kind of way.

 The linear regression seemed to evenly distribute the bias among the linearly dependent one-hot encoded categories, whereas the shrinkage models seemed to converge on feature selection that actually makes sense. Again, they all performed similarly, but the shrinkage models seem to have better explanatory power.

 I ran these models earlier this year when I was preparing for my summer internship with Los Alamos National Labs. In a future post, I will share my experiences from the internship and talk about the work I did there.