
import re
import unicodedata
import nltk
from nltk.stem.snowball import SnowballStemmer
import stanza
import spacy
from nltk.corpus import stopwords
nltk.download('stopwords')
stanza.download("es")
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import en_core_web_sm
from tqdm import tqdm

def removePunctWithRegex(text): 
  text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
  text = re.sub(r'[^\w\s\?]', ' ', text)
  return text

def removeLinesRegex(text):
  text = re.sub(r'([\;\:\|•«\n])', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def uncased(text):
  return text.lower()

def strip_accents(text):
  return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def stem(text):
  stemmer = SnowballStemmer('spanish')
  spanish_words = [stemmer.stem(t) for t in text.split()]
  return  " ".join(spanish_words)

def lemmStanza(text):
  nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma,ner')
  doc = nlp(text)
  return " ".join([word.lemma for sent in doc.sentences for word in sent.words]), [(doc.entities[i].text,doc.entities[i].type) for i in range(len(doc.entities))]

def lemmSpacy(text):
  nlp = spacy.load("es_core_news_sm")
  doc = nlp(text)
  return " ".join([sent.lemma_ for sent in doc]), [(entity.text, entity.label_) for entity in doc.ents]

def removeStopwords(text, stopwords = stopwords.words('spanish')):
  return " ".join([word for word in text.split() if word not in stopwords])


def removeSpecialChar(text):
  #removing numbers
  text = ''.join([i for i in text if not i.isdigit()])
  #removing mails
  text = re.sub('\S+@\S+','',text)
  #removing twitter user
  text = re.sub('@[^\s]+','',text)
  return text

def wordTokenize(text):
  return word_tokenize(text)

def sentenceTokenize(text):
  return nltk.sent_tokenize(text)

def text_preprocessing(text, punctuation = True, LB_whitespaces = True, normalize = True, accent = True, stemming = False, lemmatization = 'spacy', stopwords = True, custom_stopwords = None, twitter_mails_digit = False):
  if punctuation is True:
    text = removePunctWithRegex(text)
  if LB_whitespaces is True:
    text = removeLinesRegex(text)
  if normalize is True:
    text = uncased(text)
  if stopwords is True:
    text = removeStopwords(text)
  if custom_stopwords is not None:
    text = removeStopwords(text=text, stopwords = custom_stopwords)
  if accent is True:
    text = strip_accents(text)
  if stemming is True:
    text = stem(text)
  if lemmatization is 'spacy':
    text,entity = lemmSpacy(text)
  if lemmatization is 'stanza':
    text,entity = lemmStanza(text)

  if twitter_mails_digit is True:
    text = removeSpecialChar(text)
  return wordTokenize(text),entity

def corpus(dataset, punctuation = True, LB_whitespaces = True, normalize = True, accent = True, stemming = False, lemmatization = 'spacy', stopwords = True, custom_stopwords = None, twitter_mails_digit = False):
  text=[]
  entity=[]
  for j in tqdm(range(len(dataset))):
    a,b = text_preprocessing(dataset[j], punctuation, LB_whitespaces, normalize, accent, stemming, lemmatization, stopwords, custom_stopwords, twitter_mails_digit)
    if a !='' and b !='':
      text.append(a)
      entity.append(b)
  text = list(filter(lambda x: x != [], text))
  entity = list(filter(lambda x: x != [], entity))
  return text,entity