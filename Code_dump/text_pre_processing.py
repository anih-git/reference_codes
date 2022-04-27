# Preprocessing function
# Text cleaning
def text_clean(text):
  text = text.lower()  # Lowercase text
  text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1", text)
  text = re.sub(r"[^A-Za-z\s]+", "", text)
  #text = text.lower()  # Lowercase text
  clean_tokens = [t for t in text.split() if len(t) > 1]
  clean_text = " ".join(clean_tokens)
  return clean_text

# Function for text preprocessing using Spacy nlp object
def process_text(text):
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    return " ".join(result)


# Lowercase text
sample_text = "THIS TEXT WILL BE LOWERCASED. THIS WON'T: ßßß"
clean_text = sample_text.lower()
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: THIS TEXT WILL BE LOWERCASED. THIS WON'T: ßßß
# After: this text will be lowercased. this won't: ßßß

# Remove cases (useful for caseles matching)
sample_text = "THIS TEXT WILL BE LOWERCASED. THIS too: ßßß"
clean_text = sample_text.casefold()
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: THIS TEXT WILL BE LOWERCASED. THIS too: ßßß
# After: this text will be lowercased. this too: ssssss

# Remove hyperlinks
import re

sample_text = "Some URLs: https://example.com http://example.io http://exam-ple.com More text"
clean_text = re.sub(r"https?://\S+", "", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: Some URLs: https://example.com http://example.io http://exam-ple.com More text
# After: Some URLs:    More text

# Remove <a> tags but keep their content
import re

sample_text = "Here's <a href='https://example.com'> a tag</a>"
clean_text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: Here's <a href='https://example.com'> a tag</a>
# After: Here's  a tag

# Remove all HTML tags but keep their contents
import re

sample_text = """
<body>
<div> This is a sample text with <b>lots of tags</b> </div>
<br/>
</body>
"""
clean_text = re.sub(r"<.*?>", " ", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: 
# <body>
# <div> This is a sample text with <b>lots of tags</b> </div>
# <br/>
# </body>

# After: 

#  This is a sample text with lots of tags 
# Remove extra spaces, tabs, and line breaks
# You might think that the best approach to remove extra spaces, tabs, and line breaks would depend on regular expressions. But it doesn't.
# The best approach consists of using a clever combination two string methods: .split() and .join(). First, you apply the .split() method to the string you want to clean. It will split the string by any whitespace and output a list. Then, you apply the .join() method on a string with a single whitespace (" "), using as input the list you generated. This will put back together the string you split but using a single whitespace as separator.

sample_text = "     \t\tA      text\t\t\t\n\n sample       "
clean_text = " ".join(sample_text.split())
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before:      		A      text			

#  sample       
# After: A text sample

# Remove punctuation
import re
from string import punctuation

sample_text = "A lot of !!!! .... ,,,, ;;;;;;;?????"
clean_text = re.sub(f"[{re.escape(punctuation)}]", "", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: A lot of !!!! .... ,,,, ;;;;;;;?????
# After: A lot of   

# Remove numbers
import re

sample_text = "Remove these numbers: 1919191 2229292 11.233 22/22/22. But don't remove this one H2O"
clean_text = re.sub(r"\b[0-9]+\b\s*", "", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: Remove these numbers: 1919191 2229292 11.233 22/22/22. But don't remove this one H2O
# After: Remove these numbers: .//. But don't remove this one H2O

# Remove digits
sample_text = "I want to keep this one: 10/10/20 but not this one 222333"
clean_text = " ".join([w for w in sample_text.split() if not w.isdigit()]) # Side effect: removes extra spaces
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: I want to keep this one: 10/10/20 but not this one 222333
# After: I want to keep this one: 10/10/20 but not this one

# Remove non-alphabetic characters

sample_text = "Sample text with numbers 123455 and words !!!"
clean_text = " ".join([w for w in sample_text.split() if w.isalpha()]) # Side effect: removes extra spaces
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: Sample text with numbers 123455 and words !!!
# After: Sample text with numbers and words

# Remove all special characters and punctuation
import re

sample_text = "Sample text 123 !!!! Haha.... !!!! ##$$$%%%%"
clean_text = re.sub(r"[^A-Za-z0-9\s]+", "", sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: Sample text 123 !!!! Haha.... !!!! ##$$$%%%%
# After: Sample text 123  Haha

# Remove stopwords from a list
stopwords = ["is", "a"]
sample_text = "this is a sample text"
tokens = sample_text.split()
clean_tokens = [t for t in tokens if not t in stopwords]
clean_text = " ".join(clean_tokens)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: this is a sample text
# After: this sample text

# Remove short tokens
sample_text = "this is a sample text. I'll remove the a"
tokens = sample_text.split()
clean_tokens = [t for t in tokens if len(t) > 1]
clean_text = " ".join(clean_tokens)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: this is a sample text. I'll remove the a
# After: this is sample text. I'll remove the

# Transform emojis to characters
from emoji import demojize

sample_text = "I love 🥑"
clean_text = demojize(sample_text)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: I love 🥑
# After: I love :avocado:

# NLTK
# install NLTK
pip install nltk.

# Tokenize text using NLTK
from nltk.tokenize import word_tokenize

sample_text = "this is a text ready to tokenize"
tokens = word_tokenize(sample_text)
print_text(sample_text, tokens)

# ----- Expected output -----
# Before: this is a text ready to tokenize
# After: ['this', 'is', 'a', 'text', 'ready', 'to', 'tokenize']

#Tokenize tweets using NLTK
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
sample_text = "This is a tweet @jack #NLP"
tokens = tweet_tokenizer.tokenize(sample_text)
print_text(sample_text, tokens)

# ----- Expected output -----
# Before: This is a tweet @jack #NLP
# After: ['This', 'is', 'a', 'tweet', '@jack', '#NLP']

#Split text into sentences using NLTK
from nltk.tokenize import sent_tokenize

sample_text = "This is a sentence. This is another one!\nAnd this is the last one."
sentences = sent_tokenize(sample_text)
print_text(sample_text, sentences)

# ----- Expected output -----
# Before: This is a sentence. This is another one!
# And this is the last one.
# After: ['This is a sentence.', 'This is another one!', 'And this is the last one.']

# Remove stopwords using NLTK
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stopwords_ = set(stopwords.words("english"))

sample_text = "this is a sample text"
tokens = sample_text.split()
clean_tokens = [t for t in tokens if not t in stopwords_]
clean_text = " ".join(clean_tokens)
print_text(sample_text, clean_text)

# ----- Expected output -----
# Before: this is a sample text
# After: sample text

# spaCy

# Install Spacy
pip install spacy

# Get language model -- ref(https://spacy.io/models/en)
!python -m spacy download en_core_web_sm

Tokenize text using spaCy
import spacy

nlp = spacy.load("en_core_web_sm")

sample_text = "this is a text ready to tokenize"
doc = nlp(sample_text)
tokens = [token.text for token in doc]
print_text(sample_text, tokens)

# ----- Expected output -----
# Before: this is a text ready to tokenize
# After: ['this', 'is', 'a', 'text', 'ready', 'to', 'tokenize']

# Split text into sentences using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

sample_text = "This is a sentence. This is another one!\nAnd this is the last one."
doc = nlp(sample_text)
sentences = [sentence.text for sentence in doc.sents]
print_text(sample_text, sentences)

# ----- Expected output -----
# Before: This is a sentence. This is another one!
# And this is the last one.
# After: ['This is a sentence.', 'This is another one!\n', 'And this is the last one.']

# Keras
Before using Keras' snippets, you need to install the library as follows: pip install tensorflow && pip install keras.

Tokenize text using Keras
from keras.preprocessing.text import text_to_word_sequence

sample_text = 'This is a text you want to tokenize using KERAS!!'
tokens = text_to_word_sequence(sample_text)
print_text(sample_text, tokens)

# ----- Expected output -----
# Before: This is a text you want to tokenize using KERAS!!
# After: ['this', 'is', 'a', 'text', 'you', 'want', 'to', 'tokenize', 'using', 'keras']
