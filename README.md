# NLP-spanish

<div id="top"></div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#data-cleaning">Data Cleaning</a>
    </li>
    <li>
      <a href="#word-embeddings">Word Embeddings</a>
      <ul>
        <li><a href="#word2vec">Word2vec</a></li>
        <li><a href="#fastText">FastText</a></li>
        <li><a href="#glove">GloVe</a></li>
      </ul>
    </li>
    <li><a href="#topic-modeling">Topic Modeling</a></li>
    <li><a href="#text-classification">Text Classification</a>
    <ul>
        <li><a href="#sentiment-analysis">Sentiment Analysis</a></li>
      </ul>
      </li>
    <li><a href="#ner">NER</a></li>
    <li><a href="#wsp-analysis">Wsp Analysis</a>
        <ul>
        <li><a href="#topic-modeling-wsp">Topic Modeling Wsp</a></li>
        <li><a href="#word-embeddings-and-keybert">Word Embeddings and Keybert</a></li>
        <li><a href="#multiclass-classification-and-zero-shot-test">Multiclass Classification and Zero-Shot Test</a></li>
      </ul></li>
    <li><a href="#wsp-Autocorrect">Wsp Autocorrect</a></li>

  </ol>
</details>

### Prerequisites
First of all we recommend you to install the requirements.
```python
!pip3 install -r requirements.txt
```

Also, it is necessary to install some spacy models:
```python
!python -m spacy download en_core_web_sm
!python -m spacy download es_dep_news_trf
```


<!-- Data Cleaning -->
## Data Cleaning


Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect, incomplete, irrelevant, duplicated, or improperly formatted. In this file there are two scripts: `NLPSpanish_DataCleaning.ipynb` and `data_cleaning.py`. 

### The `NLPSpanish_DataCleaning.ipynb` script will divide the data cleaning in the following sections:

* Exploring the dataset
* Removing punctuation
* Removing line breaks and extra whitespaces
* Normalize text
* Removing accents
* Stemming
* Lemmatization
* Removing stopwords
* Removing twitter nicknames, emails and numbers
* Word and Sentence Tokenization

The last script is the method with all the functions defined. Here is a brief example of how to use it:

```python
import data_cleaning.spanish as sp

sentence = 'La cordillera es un sector de latinoamérica muy lindo. Tiene mucha naturaleza,\
  animales y paisajes. ¡Quiero visitarla pronto, antes de tener que irme a vivir a otro lugar! :('

cleaned_sent, entity = sp.text_preprocessing(sentence)
print(f'cleaned sentence: {cleaned_sent}')
# → cleaned sentence: ['cordillerar', 'sector', 'latinoamerico', 'lindo', 'mucho', 'naturaleza', 'animal', 'paisaje', 'querer', 'visitar','él','pronto','tener','ir','yo','vivir', 'lugar']

print(f'entity detected: {entity}')
# → entity detected: [('cordillera sector latinoamerica', 'LOC')]
```
Also you can work with corpus i.e. with tokenized sentences 
```python
document = sp.sentenceTokenize(sentence)
print(f'corpus: {document}')
# → corpus: ['La cordillera es un sector de latinoamérica muy lindo.', 'Tiene mucha naturaleza, animales y paisajes.', '¡Quiero visitarla pronto, antes de tener que irme a vivir a otro lugar!', ':(']

cleaned_doc,entities = sp.corpus(dataset=document)
print(f'cleaned corpus: {cleaned_doc}')
# → cleaned corpus: [['cordillerar', 'sector', 'latinoamerico', 'lindo'], ['mucho', 'naturaleza', 'animal', 'paisaje'], ['querer', 'visitar', 'él', 'pronto', 'tener', 'ir', 'yo', 'vivir', 'lugar']]
print(f'entities detected: {entities}')
# → entities detected: [[('cordillera sector latinoamerica lindo', 'LOC')]]
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Word Embeddings-->
## Word Embeddings

Working with machine learning models means that you must expressed numerically the information that you want to process. Therefore, there are many algorithms that enable words to be expressed mathematically, such as Bag-Of-Word, TF-IDF, Word2Vec, FastText. This is called feature extraction or feature encoding.
In this archive you will find a brief introduction to feature encoding methods using a spanish twitter dataset.

<!-- Word2vec-->

### Word2vec

With `w2v.ipynb` you will be able to understand Skip-gram and Continuous Bag-of-Words method. Also an example of how to train a model and how to use a pretrained W2V.
<!-- FastText-->
### FastText

This word embedding uses a Skip-gram method, in `fastText.ipynb` you can find a training example and a pretrained model.
<!-- GloVe-->
### GloVe

`GloVe.ipynb` has a spanish pretrained model example.

<p align="right">(<a href="#top">back to top</a>)</p>


<!--topic-modeling-->
## Topic Modeling

Topic modeling is a NLP task that consists in recognizing the words from the topics present in the documents. In this section we use only unsupervised methods, there are two scripts: `topic_modeling.ipynb` and `topic_m.py`. The first script is a tutorial of this task.

### The `topic_m.py` presentd many ways to visualize the results:

* Bar chart word count and weights of each topic keyword
* A fast pyLDAvis visualization
* Dataframe with the ppl vocabulary and score per topic
* Dataframe with the dominant topic and its percentage contribution in each document
* Dataframe with the most representative sentence for each topic
* Sentence chart colored by topic
* TSNE interactive topic clustering

Here is an example of how to use the script and the expected output:
```python
import topic_m as tm

model_list, coherence_values = tm.compute_coherence_values(dictionary=id2word, corpus=corpus, id2word=id2word, texts=cleaned_doc_train, start=1, limit=10, step=2, model_='LDA')
lda_model = model_list[coherence_values.index(max(coherence_values))]
tm.tsne_plot(model_=lda_model,corp=corpus,name_model='LDA',text=cleaned_doc_train,interactive_labels = True)

```
![](/tm_ex.png)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Text-Classification -->
## Text Classification

Text classification is the process of categorizing the text into a group of words. In the following scripts were tested machine and deep learning models such as LSTM, Naive Bayes, Transformers, Random Forest, etc. The Notebook `TC_complaints_classifiers.ipynb` looks for the best model (using cross-validation methods) for a translated Complains Dataset.


### Sentiment Analysis
An especific case of text classification is sentiment analysis. This focuses on the polarity of a text (positive, negative, neutral). For `TC_sentiment_analysis.ipynb` we used a Hate Speech Spanish Dataset.
Models trained:
*  LSTM
*  Support-vector machin
*  Random Forest Classifier
*  Logistic Regression
*  Naive Bayes
*  Finetuned Transformer
<p align="right">(<a href="#top">back to top</a>)</p>
<!-- ner-->

## NER

In `ner.ipynb` we show you two differents ways for recognizing entities. 
* Using a pretrained transformer from Hugging Face
* Using the `data_cleaning.ipynb` functions
There are not trained models because NER's dataset are expensive.
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- wsp-Analysis-->
## Wsp Analysis

We explored some ideas with the WhatsApp Dataset.

### Topic Modeling Wsp

Testing LSI and LSA, in `topic_modeling_wsp.ipynb` we extracted the topics of the most popular group chat.

### Word Embeddings and Keybert
In `word_embeddings_keybert.ipynb` there are two word embeddings trained with the wsp messages; Word2Vec and FastText. Both were tested with the KeyBert arquitecture. 

### Multiclass Classification and Zero-Shot Test
There are just 550 labeled rows even though it is a small amount of labeled data, we experimented with zero-shot learning trained transformers and then with multiclass classification methods. Hopefully, in some future we expect to be able to guide the data labeling process.
<p align="right">(<a href="#top">back to top</a>)</p>

## Wsp Autocorrect
We search and experiment with some autocorrect techniques. `autocorrect_datasetwsp.ipynb` contains a autocorrect method using FastText. Also, the `palabras_chilenas.csv` and `corrected_with_speller.csv` datasets were created.
