
# Natural Language Processing 

__NLP uses:__

- Sentiment analysis
- Predict genre of book
- Question answering
- Machine translator or speech recognition

__Librairies:__ Spacy, NLTK ...

__Bag of Words__:

Very popular NLP model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

It involves two things:

- A vocabulary of known words
- A measure of the presence of known words

In this section, we will understand and learn how to:

- Cleans text to prepare them for machine learning models
- Create a Bag of words model
- Apply machine learning models onto this bag of worlds model.

## Practical Example

The dataset contains reviews of a restaurant and the goal is to separate the good and bad reviews.

The tsv format is use because the tab separator in our context use is the best to use (a comma will create other columns).



```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset - Quoting for ingnoring double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Text cleaning to Bag of words

The goal is to only get the relevant words and avoid punctuation, numbers, capitals or stop words and also apply stemming to get the root of a word and avoid different version of a word.

At the end we will apply the tokenization process to create our bag of words by splitting the text to a matrix of words (in columns).

---

## __Step 1:__ Only keeping the letters and remove punctuation and numbers

Using regex and sub method using a regular expression `^a-zA-Z]`

```python
import re

# adding the space to replace the removed characters
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
```


```python
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
print("Before : {}".format(dataset['Review'][0]))
print("After : {}".format(review))
```

    Before : Wow... Loved this place.
    After : Wow    Loved this place 


---

## __Step 2:__ Putting all the letters to lowercase
```python
#To lowercase
review = review.lower()
```


```python
review_lower = review.lower()

print("Putting all the letters to lowercase...\n")
print("Before : {}".format(re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])))
print("After : {}".format(review_lower))
```

    Putting all the letters to lowercase...
    
    Before : Wow    Loved this place 
    After : wow    loved this place 


---

## __Step 3:__ Remove the non significant words (stopwords)

If we are using the first line we see that words like `this` is not really usefull for machine learning algo.

To remove the stopwords we are going to use yhe __nltk__ library and its stopwords list.

For each review we will split the text in several words and check if each of the words are in the stopwords list.

```python
import nltk

# Importing and dowload the list of useless words
nltk.download('stopwords')

#split the text
review = review.split()

# removing stop words - set function is used find faster stop words matches
review = [word for word in review if not word in set(stopwords.words('english'))]

```


```python
review_splitted = review.split()

print("Splitting text...\n")

print("Before : {}".format(re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]).lower()))
print("After : {}".format(review_splitted))
```

    Splitting text...
    
    Before : wow    loved this place 
    After : ['Wow', 'Loved', 'this', 'place']



```python
from nltk.corpus import stopwords

review_no_stop_words = [word for word in review_splitted if not word in set(stopwords.words('english'))]
print("Removing stopwords...\n")
print("Before : {}".format(review_splitted))
print("After : {}".format(review_no_stop_words))
```

    Removing stopwords...
    
    Before : ['Wow', 'Loved', 'this', 'place']
    After : ['Wow', 'Loved', 'place']


---

## __Step 4:__ Stemming

To get the root of a word to avoid the different versions.

```python
# importing PorterStemmer class
from nltk.stem.porter import PorterStemmer

# Create an object of the Porter stemmer
ps = PorterStemmer()

# Applying steamer to our list of words
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

# convert our list to text
review = ' '.join(review)

```



```python
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

review_stemmed_no_stop_words = [ps.stem(word) for word in review_no_stop_words if not word in set(stopwords.words('english'))]
print("Stemming ...\n")
print("Before : {}".format(review_no_stop_words))
print("After : {}".format(review_stemmed_no_stop_words))
```

    Stemming ...
    
    Before : ['Wow', 'Loved', 'place']
    After : ['wow', 'love', 'place']



```python
review_text_stemmed_no_stop_words = ' '.join(review_stemmed_no_stop_words)

print("Recreating text on each line ...\n")
print("Before : {}".format(review_stemmed_no_stop_words))
print("After : {}".format(review_text_stemmed_no_stop_words))
```

    Recreating text on each line ...
    
    Before : ['wow', 'love', 'place']
    After : wow love place


---

## Full code for cleaning process (for each review)
 


```python
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    # appending clean review to our list of common words 
    corpus.append(review)
    
print("************* Before cleaning... ************* \n ")
print(' \n\n'.join(dataset['Review'][:10].tolist()))
print("\n\n\n")
print("************* After cleaning ... ************* \n")
print(' \n\n'.join(corpus[:10]))
```

    [nltk_data] Downloading package stopwords to /Users/yanni-benoit-
    [nltk_data]     iyeze/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


    ************* Before cleaning... ************* 
     
    Wow... Loved this place. 
    
    Crust is not good. 
    
    Not tasty and the texture was just nasty. 
    
    Stopped by during the late May bank holiday off Rick Steve recommendation and loved it. 
    
    The selection on the menu was great and so were the prices. 
    
    Now I am getting angry and I want my damn pho. 
    
    Honeslty it didn't taste THAT fresh.) 
    
    The potatoes were like rubber and you could tell they had been made up ahead of time being kept under a warmer. 
    
    The fries were great too. 
    
    A great touch.
    
    
    
    
    ************* After cleaning ... ************* 
    
    wow love place 
    
    crust good 
    
    tasti textur nasti 
    
    stop late may bank holiday rick steve recommend love 
    
    select menu great price 
    
    get angri want damn pho 
    
    honeslti tast fresh 
    
    potato like rubber could tell made ahead time kept warmer 
    
    fri great 
    
    great touch


---

## Step 5: Creating the bag of words model

The bag of words model is used after creating the corpus and to create it you have to take all disctinct words and create on column for each word (__tokenization__). 
That will create a matrix of the reviews and word in column, if a line contain a word there will be a 1 and 0 instead.

We need to create this model to use a machine learning model because the ml model will be trained on the reviews and help it to understand the correlation (review <-> word) for the classification : __we use independent variable (word) to predict dependent variable (good or bad).__

We use the [__CountVectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) method of the __sklearn library__ to reproduce the tokenization process

We already cleaned the text before so we don't need to use every parameters. 
That's should be more useful if you wand to apply more cleaning steps for complicated text like scrapped text.


```python
# importing the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer

# Creating CountVectorizer object - MaxFeatures is used to remove non relevant words (we have 1500 column).
cv = CountVectorizer(max_features = 1500)

# Fitting our corpus using tokenisation - Matrix of features or independent variable
X = cv.fit_transform(corpus).toarray()

#Selecting dependent variable from dataset - Reviews likes
y = dataset.iloc[:, 1].values
```


```python
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

print("************* Matrix of feature... ************* \n ")
print(X[:5])

print("************* Independant variable... ************* \n ")

print(y[:5])
```

    ************* Matrix of feature... ************* 
     
    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    ************* Independant variable... ************* 
     
    [1 0 0 1 1]


---

## Step 6: Applying a Machine Learning Model

The most common models used for NLP are Naive Bayes, Decision Trees and Random forest models.

For our example we will try the Naive Bayes model.

```python

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

```


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("************* Prediction... ************* \n ")
print(y_pred[:5])

# Comparing Predictions with Test set
print("************* Results... ************* \n ")

results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
l = []
for index, row in results.iterrows():
    if row['y_test'] == row['y_pred']:
        result = 'Yes'
    else:
        result = 'No'
    l.append(result)
results['is ok ?'] = l


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("************* Confusion Matrix... ************* \n ")
cm


```

    ************* Prediction... ************* 
     
    [1 1 1 0 0]
    ************* Results... ************* 
     
    ************* Confusion Matrix... ************* 
     





    array([[55, 42],
           [12, 91]])



We got 55 + 91 good results so 73%

## Homework challenge 
```
Hello students,

congratulations for having completed Part 7 - Natural Language Processing.

If you are up for some practical activities, here is a little challenge:

1. Run the other classification models we made in Part 3 - Classification, other than the one we used in the last tutorial.

2. Evaluate the performance of each of these models. Try to beat the Accuracy obtained in the tutorial. But remember, Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall). Please find below these metrics formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

3. Try even other classification models that we haven't covered in Part 3 - Classification. Good ones for NLP include:

    CART
    C5.0
    Maximum Entropy

```


```python

```
