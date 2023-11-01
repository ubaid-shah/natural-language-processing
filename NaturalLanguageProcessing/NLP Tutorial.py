#!/usr/bin/env python
# coding: utf-8

# In[1]:


# create a string
amount = u"₹50"
print('Default string: ', amount, '\n', 'Type of string', type(amount), '\n')

# encode to UTF-8 byte format
amount_encoded = amount.encode('utf-8')
print('Encoded to UTF-8: ', amount_encoded, '\n', 'Type of string', type(amount_encoded), '\n')


# sometime later in another computer...
# decode from UTF-8 byte format
amount_decoded = amount_encoded.decode('utf-8')
print('Decoded from UTF-8: ', amount_decoded, '\n', 'Type of string', type(amount_decoded), '\n')


# In[1]:


# import the regular expression module
import re

# input string on which to test regex pattern
string = 'The roots of education are bitter, but the fruit is sweet.'


# In[3]:


# regex pattern to check if 'education' is present in a input string or not.
pattern = re.search("education?",string).group() # write regex to extract 'education'


# In[4]:


# check whether pattern is present in string or not
result = re.search(pattern, string)


# In[5]:


# evaluate result - don't change the following piece of code, it is used to evaluate your regex
if result != None:
    print(True)
else:
    print(False)


# In[6]:


result


# In[7]:


pattern


# In[8]:


result.start()    # to find the index of start of pattern


# In[10]:


# store the end of the match using result.end()
result.end()


# In[11]:


# to find the number of characters that has been matched from pattern
result.end()-result.start()


# In[12]:


text=["The tree stands tall.","There are a lot of trees in the forest.",
      'The boy is heading for the school.',"It's really hot outside!"]


# In[13]:


len(text)


# In[23]:


pattern="tree?"
list1 = []
for sentences in text:
    list1.append(re.search(pattern,sentences))


# In[24]:


list1


# In[33]:


pattern1='xyz*'
result1=re.search(pattern1,'xyyyz')
print(result1)


# In[38]:


pattern ='10+' # write your pattern here

# check whether pattern is present in string or not
result = re.search(pattern,'1000')


# In[39]:


result


# In[51]:


pattern ='awesome{3,}' # write your regex pattern here

# check whether pattern is present in string or not
result = re.search(pattern, "awesomeeeeeee")
print(result)


# In[41]:


result


# In[58]:


pattern = '(23)+[0-9]*(78)+' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "2323233787878")

print(result)


# In[61]:


# Write a regular expression that matches the following strings: 
# Basketball 
# Baseball 
# Volleyball 
# Softball 
# Football


# regex pattern
pattern = '(Basket|Base|Volley|Soft|Foot|Cricket)(B|b)all' # write your egex here

# check whether pattern is present in string or not
result = re.search(pattern, "Cricketball")
result1 = re.search(pattern, "CricketBall")


print(result,"\n",result1)


# In[66]:


pattern = '[0-9]*[a-z]*\*' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "4*5*6=120")

print(result)


# In[67]:


pattern = '\*' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "4*5*6=120")

print(result)


# In[68]:


pattern = '[0-9]*[a-z]*\*' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "4a*5b*6c=120")

print(result)


# In[4]:


'''Anchors
Description
Write a pattern that matches all the dictionary words that start with ‘A’

Positive matches (should match all of these):
Avenger
Acute
Altruism

Negative match (shouldn’t match any of these):
Bribe
10
Zenith '''


# regex pattern
pattern = '^a' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "Avengers", re.I)  # re.I ignores the case of the string and the pattern
result1 = re.search(pattern, "Bribe", re.I)
print("result: " ,result)
print("result1: ",result1)


# In[5]:


""" Write a pattern which matches a word that ends with ‘ing’.
Words such as ‘playing’, ‘growing’, ‘raining’, etc. should match while words that don’t have ‘ing’ at the end shouldn’t match.
"""


# regex pattern
pattern = '(ing)$' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern,"string")  
result1 = re.search(pattern,"strig")  

print("result: " ,result)
print("result1: ",result1)


# In[6]:


"""
Write a regular expression that matches any string that starts with one or more ‘1’s, 
followed by three or more ‘0’s, 
followed by any number of ones (zero or more), 
followed by ‘0’s (from one to seven), 
and then ends with either two or three ‘1’s.
"""

pattern = '^1+0{3,}1*0{1,7}1{2,3}$' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, "100100000100001")

print("result: " ,result)


# In[9]:


'''
Write a regular expression to match first names (consider only first names, i.e. there are no spaces in a name) 
that have length between three and fifteen characters.

Sample positive match:
Amandeep
Krishna

Sample negative match:
Balasubrahmanyam
'''
string="Balasubrahmanyam"
pattern = '.{3,14}' # write your regex here

# check whether pattern is present in string or not
result = re.search(pattern, string)

print(result)


# In[10]:


'''
Meta-sequences
Description
Write a regular expression with the help of meta-sequences that matches usernames of the users of a database. 

The username starts with alphabets of length one to ten characters long and then followed by a number of length 4.

Sample positive matches:
sam2340 
irfann2590 

Sample negative matches:
8730 
bobby9073834 
sameer728 
radhagopalaswamy7890
'''


# regex pattern
pattern = '^[a-z]{1,10}[\d]{4}$'

# check whether pattern is present in string or not
result = re.search(pattern, "irfann2590", re.I)
print(result)


# In[14]:





string= '<html> <head> <title> My amazing webpage </title> </head> <body> Welcome to my webpage! </body> </html>' 



# regex pattern
pattern = '<.*>'

# check whether pattern is present in string or not
result = re.search(pattern, string, re.M)  # re.M enables tha tpettern to be searched in multiple lines

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
if (result != None) and (len(result.group()) > 6):
    print(True)
else:
    print(False)
    
print (result)    


# In[15]:


string= '<html> <head> <title> My amazing webpage </title> </head> <body> Welcome to my webpage! </body> </html>' 



# regex pattern
pattern = '<.*?>'

# check whether pattern is present in string or not
result = re.search(pattern, string, re.M)  # re.M enables tha tpettern to be searched in multiple lines

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
if (result != None) and (len(result.group()) > 6):
    print(True)
else:
    print(False)
    
print (result)    


# In[12]:


print(re.match("b+", "absurd"))
print(re.match("b+", "britain"))
print(re.match("b+.b*..", "bulb"))


# **re.match()** returns a non-empty match only if the match is present at the very beginning of the string. The pattern is present in the string right at the start.

# The **re.sub()** function is used to substitute a part of your string using a regex pattern. It is often the case when you want to replace a substring of your string where the substring has a particular pattern that can be matched by the regex engine and then it is replaced by the **re.sub()** command. 

# In[13]:


pattern = "\d"
replacement = "X"
string = "My address is 13B, Baker Street"

re.sub(pattern, replacement, string)


# In[14]:


string="Pink is very good clour."
re.sub("clour", "colour", string)


# In[30]:


# You are given the following string: 

# “You can reach us at 07400029954 or 02261562153 ”
 
# Substitute all the 11-digit phone numbers present in the above string with “####”. 




# regex pattern
pattern = '[0-9]{11}' # write a regex that detects 11-digit number

# replacement string
replacement = '####' # write the replacement string

# check whether pattern is present in string or not
result = re.sub(pattern,replacement,"12345678901")
result


# In[31]:


# You are given the following string: 

# “You can reach us at 07400029954 or 02261562153 ”
 
# Substitute all the 11-digit phone numbers present in the above string with “####”. 




# regex pattern
pattern = '\d{11}' # write a regex that detects 11-digit number

# replacement string
replacement = '####' # write the replacement string

# check whether pattern is present in string or not
result = re.sub(pattern,replacement,"1234567890")
result


# In[36]:


# Write a regular expression such that it replaces the first letter of any given string with ‘$’. 

# For example, the string ‘Building careers of tomorrow’ should be replaced by “$uilding careers of tomorrow”.

string= 'Building careers of tomorrow'
pattern = '^.' # write a regex that detects the first character of a string

# replacement string
replacement = '$' # write the replacement string

# check whether pattern is present in string or not
result = re.sub(pattern,replacement,string)  # pass the parameters to the sub function

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
print(result[0] == '$')

print(result)


# In[37]:


## Example usage of finditer(). Find all occurrences of word Festival in given sentence

text = 'Do not compare apples with oranges. Compare apples with apples'
pattern = '(.){5,}'
for match in re.finditer(pattern, text):
    print('START -', match.start(), end="")
    print('END -', match.end())


# In[42]:


import re
import ast, sys
string = 'Do not compare apples with oranges. Compare apples with apples'

# regex pattern
pattern = "\w+"

# store results in the list 'result'
result = []

# iterate over the matches
for match in re.finditer(pattern,string): # replace the ___ with the 'finditer' function to extract 'pattern' from the 'string'
    if len(match.group()) >= 5:
        result.append(match)
    else:
        continue

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
print(len(result))
result


# In[54]:


# Write a regular expression to extract all the words that have the suffix ‘ing’ using the re.findall() function. 
# Store the matches in the variable ‘results’ and print its length.

Sample= "Playing outdoor games when its raining outside is always fun!"


pattern = '\w+ing' # write regex to extract words ending with 'ing'

# store results in the list 'result'
result =re.findall(pattern,Sample) # extract words having the required pattern, using the findall function

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
print(len(result))
result


# In[56]:


Sample= "Playing outdoor games when its raining outside is always fun!"
pattern = '\w+e.'
result =re.findall(pattern,Sample)
result


# In[57]:


Sample= "Playing outdoor games when its raining outside is always fun!"
pattern = '\w+!'
result =re.findall(pattern,Sample)
result


# In[60]:


Sample= "Playing outdoor games when its raining outside is always fun!"
pattern = re.compile('\w+!')
result =pattern.findall(Sample)
result


# In[63]:


# You have a string which contains a data in the format DD-MM-YYYY. Write a regular expression to extract 
# the date from the string.

# Sample input: "Today’s date is 18-05-2018."

# Sample output: 18-05-2018

string="Today’s date is 18-05-2018."

pattern = '(\d{2})-(\d{2})-(\d{4})' # write regex to extract date in DD-MM-YYYY format

# store result
result = re.search(pattern,string).group() 

result


# In[3]:


# Write a regular expression to extract the domain name from an email address. 
# The format of the email is simple - the part before the ‘@’ symbol contains alphabets, numbers and underscores. 
# The part after the ‘@’ symbol contains only alphabets followed by a dot followed by ‘com’ 
 
# Sample input: 
# user_name_123@gmail.com 
 
# Expected output: 
# gmail.com


string="user_name_123@gmail.com"
# regex pattern
pattern = '(@)(\w+\.com)' # write regex to extract email and use groups to extract domain name ofthe mail

# store result
result = re.search(pattern, string)

# extract domain using group command
if result != None:
    domain = result.group(2) # use group to extract the domain from result
else:
    domain = "NA"

# evaluate result - don't change the following piece of code, it is used to evaluate your regex
print(domain)


# In[4]:


# items contains all the files and folders of current directory
items = ['photos', 'documents', 'videos', 'image001.jpg','image002.jpg','image005.jpg', 'wallpaper.jpg',
         'flower.jpg', 'earth.jpg', 'monkey.jpg', 'image002.png']

# create an empty list to store resultant files
images = []

# regex pattern to extract files that end with '.jpg'
pattern = ".*\.(jpg|png)$"

for item in items:
    if re.search(pattern, item):
        images.append(item)

# print result
print(images)


# In[5]:


# items contains all the files and folders of current directory
items = ['photos', 'documents', 'videos', 'image001.jpg','image002.jpg','image005.jpg', 'wallpaper.jpg',
         'flower.jpg', 'earth.jpg', 'monkey.jpg', 'image002.png']

# create an empty list to store resultant files
images = []

# regex pattern to extract files that start with 'image' and end with '.jpg'
pattern = "image.*\.jpg$"

for item in items:
    if re.search(pattern, item):
        images.append(item)

# print result
print(images)


# In[6]:


# You learnt how to extract and plot word frequencies from a list of words. In this exercise,
# you need to extract the third most frequent word of a book (the book is provided) and print it's frequency.


import requests
from nltk import FreqDist

# load the ebook
url = "https://www.gutenberg.org/files/16/16-0.txt"
peter_pan = requests.get(url,verify=False)

# break the book into different words using the split() method
peter_pan_words = peter_pan.text.split() # write your code here

# build frequency distribution using NLTK's FreqDist() function
word_frequency = FreqDist(peter_pan_words) # write your code here

# extract the frequency of third most frequent word
freq = word_frequency.most_common(3)[2][1]

# print the third most frequent word - don't change the following code, it is used to evaluate the code
print(freq)


# In[7]:


word_frequency


# In[8]:


freq1 = word_frequency.most_common(3)
freq1


# In[9]:


freq2 = word_frequency.most_common(3)[2]
freq2


# In[10]:


freq3 = word_frequency.most_common(3)[2][1]
freq3


# In[12]:


# In this exercise, you'll remove stop words in a given corpus of text of a book. 
# Then, you'll print the frequency of the most frequent word.





import requests
from nltk import FreqDist
from nltk.corpus import stopwords

# load the ebook
url = "https://www.gutenberg.org/files/16/16-0.txt"
peter_pan = requests.get(url,verify=False).text

# break the book into different words using the split() method
peter_pan_words = peter_pan.split()

# build frequency distribution using NLTK's FreqDist() function
word_frequency = FreqDist(peter_pan_words)

# extract nltk stop word list
stopwords = stopwords.words('english')

# remove 'stopwords' from 'peter_pan_words'
no_stops = [word for word in peter_pan_words if word not in stopwords] # write code here

# create word frequency of no_stops
word_frequency = FreqDist(no_stops) # write code here

# extract the most frequent word and its frequency
frequency = word_frequency.most_common(1)[0][1]

# print the third most frequent word - don't change the following code, it is used to evaluate the code
print(frequency)


# In[3]:


url1="https://cdn.upgrad.com/UpGrad/temp/bab3e784-e601-4911-9000-f1fbc994a62d/SMSSpamCollection.txt"


# In[7]:


url2= "https://cdn.upgrad.com/UpGrad/temp/a4964625-11c7-4043-adc5-23c0160b2ac1/SMSSpamCollection.txt"


# In[8]:


import urllib.request    
urllib.request.urlretrieve(url2, "bagsofwords.txt")


# In[5]:


import nltk
nltk.download('punkt')


# In[19]:


from nltk.tokenize import word_tokenize
import ast, sys
sentence = "I Love Pasta!"

# tokenise sentence into words
words = word_tokenize(sentence)# write your code here

# print length - don't change the following piece of code
print(len(words))


# In[20]:


# Write a piece of code that breaks a given sentence into words and stores them in a list. 
# Then remove the stop words from this list and then print the length of the list. Again, use the NLTK tokeniser to do this.

# Sample input: 
# “Education is the most powerful weapon that you can use to change the world” 

# Expected output: 
# 6





from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ast, sys
sentence = 'Education is the most powerful weapon that you can use to change the world'

# change sentence to lowercase
sentence = sentence.lower() # write code here

# tokenise sentence into words
words = word_tokenize(sentence) # write code here

# extract nltk stop word list
stopwords = stopwords.words('english') # write code here

# remove stop words
no_stops = [word for word in words if word not in stopwords] # write code here

# print length - don't change the following piece of code
print(len(no_stops))


# In[22]:


# Write a Python code using the NLTK library that breaks a given piece of text containing 
# multiple sentences into different sentences. Finally print the total number of sentences in the text.

# Sample input: 
# Develop a passion for your learning. If you do, you’ll never cease to grow.

# Expected output:

# 2 

from nltk.tokenize import sent_tokenize

sentence = 'Develop a passion for your learning. If you do, you’ll never cease to grow.'

# change sentence to lowercase
sentence = sentence.lower() # write code here

# tokenise sentence into words
sentence = sent_tokenize(sentence) # write code here


# print length - don't change the following piece of code
print(len(sentence))


# In[26]:


# Description
# Use NLTK’s regex tokeniser to extract all the mentions from a given tweet and 
# then print the total number of mentions. A mention comprises of a ‘@’ symbol followed 
# by a username containing either alphabets, numbers or underscores.

# Sample tweet:
# So excited to be a part of machine learning and artificial intelligence program made by @upgrad and @iiitb

# Expected output:
# 2 (because there are two mentions - ‘@upgrad’ and ‘@iiitb’ )



from nltk.tokenize import regexp_tokenize 
from nltk.corpus import stopwords
import ast, sys
text = 'So excited to be a part of machine learning and artificial intelligence program made by @upgrad and @iiitb'

# change text to lowercase
text = text.lower() # write code here

# pattern to extract mentions
pattern = "@\w+" # write regex pattern here

# extract mentions by using regex tokeniser
mentions = regexp_tokenize(text,pattern) # write code here

# print length - don't change the following piece of code
print(len(mentions))
mentions


# In[9]:


url3= "https://www.msn.com/en-in/entertainment/other/lata-mangeshkar-s-death-breaks-bollywood-akshay-kumar-kangana-ranaut-sonu-sood-and-ocean-of-celebs-mourn-demise/ar-AATw17k?ocid=msedgntp"


# In[10]:


urllib.request.urlretrieve(url3, "msn_news.txt")


# # Bag of Words

# In[2]:


# load all necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth', 100)


# In[6]:


# load data
spam = pd.read_csv("bagsofwords.txt", sep = "\t", names=["label", "message"])
spam.head()


# In[7]:


spam = spam.iloc[0:100,:]
print(spam,spam.shape)


# In[8]:


# extract the messages from the dataframe
messages = spam.message
print(messages)


# In[9]:


# convert messages into list
messages = [message for message in messages]
print(messages)


# In[10]:


def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # join words to make sentence
    document = " ".join(words)
    
    return document


# In[11]:


# preprocess messages using the preprocess function
messages = [preprocess(message) for message in messages]
print(messages)


# In[12]:


# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)


# In[16]:


print(bow_model)


# In[13]:


# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names())


# In[17]:


size=vectorizer.get_feature_names()


# In[18]:


print(size)


# In[21]:


len(size)


# In[ ]:





# # Stemming

# In[8]:


import re
import ast, sys
word = "Playing"

# create function to chop off the suffixes 'ing' and 'ed'
def stemmer(word):
    if re.search('(ing|ed)$',word):
        word= re.sub('(ing|ed)$',"",word) # write your code here   
    return word

# stem word -- don't change the following code, it is used to evaluate your code
print(stemmer(word))

print(word)


# In[11]:


print(stemmer("employied"))


# In[13]:


# Use Porter stemmer to stem a given word

# Sample input:
# Gardening

# Expected output:
# Garden


from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import ast, sys
word = "Gardening"

# instantiate porter stemmer
stemmer = PorterStemmer() # write code here

# stem word
stemmed = stemmer.stem(word) # write your code here

# print stemmed word -- don't change the following code, it is used to evaluate your code
print(stemmed)


# In[14]:


# Stem a given word using Snowball stemmer.

# Sample input:
# coming

# Expected output:
# come


from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import ast, sys
word = "coming"

# instantiate porter stemmer
stemmer = SnowballStemmer("english") # write code here

# stem word
stemmed = stemmer.stem(word) # write code here

# print stemmed word -- don't change the following code, it is used to evaluate your code
print(stemmed)


# In[18]:


import nltk
nltk.download('wordnet')


# In[20]:


from nltk.stem import WordNetLemmatizer
import ast, sys
word = "schooling"

# instantiate wordnet lemmatizer
lemmatizer = WordNetLemmatizer() # write code here

# lemmatize word
lemmatized = lemmatizer.lemmatize(word,pos="v") # write code here. Pass the parameter -> pos='v' to the lemmatize function to lemmatize verbs correctly.

# print lemmatized word -- don't change the following code, it is used to evaluate your code
print(lemmatized)


# In[18]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# consider the following set of documents
documents = ["The coach lumbered on again, with heavier wreaths of mist closing round it as it began the descent.",
             "The guard soon replaced his blunderbuss in his arm-chest, and, having looked to the rest of its contents, and having looked to the supplementary pistols that he wore in his belt, looked to a smaller chest beneath his seat, in which there were a few smith's tools, a couple of torches, and a tinder-box.",
            "For he was furnished with that completeness that if the coach-lamps had been blown and stormed out, which did occasionally happen, he had only to shut himself up inside, keep the flint and steel sparks well off the straw, and get a light with tolerable safety and ease (if he were lucky) in five minutes.",
            "Jerry, left alone in the mist and darkness, dismounted meanwhile, not only to ease his spent horse, but to wipe the mud from his face, and shake the wet out of his hat-brim, which might be capable of holding about half a gallon.",
            "After standing with the bridle over his heavily-splashed arm, until the wheels of the mail were no longer within hearing and the night was quite still again, he turned to walk down the hill."]


# preprocess document
def preprocess(document):
    'changes document to lower case, removes stopwords and stems words'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    # stem
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # join words to make sentence
    document = " ".join(words)
    
    return document











# In[5]:


# preprocess documents using the preprocess function and store the documents again in a list
documents = [preprocess(document) for document in documents] # write code here
print(documents)


# In[6]:


# create tf-idf matrix
## write code here ##

vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(documents)
print(tfidf_model)


# In[16]:


# extract score
score =tfidf_model[2,14] # replace -1 with the score of 'belt' in document two. You can manually write the value by looking at the tf_idf model


# In[17]:


# print the score -- don't change the following piece od code, it's used to evaluate your code
print(round(score, 4))


# In[ ]:




