## NLTK tutorial (2018)
자연어처리에서의 전처리 방법을 강의를 듣고 정리한 내용  

강의 : <https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/>

  
- nltk에서 필요한 패키지를 다운받는 방법
```python
import nltk
nltk.download() 
```

<br/>

-------
### 1. Tokenizing Words and Sentences with NLTK

- sent_tokenize : 문장별로 분리 / nltk가 Ms.smith 이런걸 잡아낸다
- word_tokenize : 단어별로 분리 / space로 split

```python
from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Hello Mr.smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkis-blue. You should not eat cardboard."

# sent_tokenize
print(sent_tokenize(example_text))

# word_tokenize
print(word_tokenize(example_text))

# 단어 하나하나 출력
for i in word_tokenize(example_text):
    print(i)
```


##### 실행결과
```
# sent_tokenize 
['Hello Mr.smith, how are you doing today?', 'The weather is great and Python is awesome.', 'The sky is pinkis-blue.', 'You should not eat cardboard.']
```
```
# word_tokenize
['Hello', 'Mr.smith', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', 'and', 'Python', 'is', 'awesome', '.', 'The', 'sky', 'is', 'pinkis-blue', '.', 'You', 'should', 'not', 'eat', 'cardboard', '.']
```
```
# 단어 하나하나 출력
Hello
Mr.smith
,
how
are
you
doing
today
?
  
...
  
You
should
not
eat
cardboard
.
```
<br/>

-------

### 2. Stop words with NLTK
- stopwords : '불용어'라고 하는데, 이는 문장 내에서 많이 등장하지만 그 단어 자체가 큰 의미를 가지고 있지 않기 때문에 분석에 도움이 되지 않는 단어들을 뜻한다. 
- 예를 들어, 'the','of','you','is' 와 같은 단어들은 문장 내에서 발생하는 빈도가 매우 높지만 실제 분석에서는 중요하지 않을 확률 또한 높다. 그렇기 때문에 전처리 과정에서 불용어를 제거할 수 있다.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words=set(stopwords.words("english"))
print(stop_words)
```


- nltk는 영어에서 아래와 같은 단어들을 stopwords로 정의하고 있다.
```
# print(set(stopwords.words("english")))
{'than', 'such', 'been', 'ours', 'at', 'm', 'y', 'again', 'you', 'themselves', 'about', 'between', "wasn't", 'with', 'mightn', 'only', 'up', "mustn't", 'i', 'are', "should've", 'which', 'more', 'couldn', 'yours', "hadn't", "hasn't", "don't", "isn't", 'he', 'above', 'now', 'until', 'were', 'in', "you've", 'll', 'to', 'this', 'weren', 'have', 'most', 'those', 'does', 'who', 'there', 'ain', "wouldn't",

'over', 'some', 'whom', "couldn't", "you're", 'these', 'isn', 'other', 'during', 'through', 'ma', 'once', 'after', 'his', "aren't", 'same', 'she', 'being', 'doing', 'below', 'or', 'and', 'we', 'before', 'all', 'mustn', 'do', "shan't", 'not', 'did', 'any', 'd', 'from', "needn't", 'but', 'too', 'haven', "she's", 'then', 'its', 'because', 'didn', "it's", 'don', "doesn't", 'further', 'each', 'nor',

'while', 'just', 'of', 'my', 'ourselves', 'the', 'by', 'aren', 'will', 'for', 'as', 'him', 'itself', 'them', 'few', 'should', 'why', 'an', 'off', 'her', 'down', 't', 'their', 'himself', 'is', 'own', "won't", 'hadn', 'me', "shouldn't", 'won', 'where', 'so', 'how', 'myself', 'here', 'very', 've', "mightn't", 'hers', "that'll", 're', 'wasn', "you'll", 'both', 'what', 's', 'yourselves', 'shan', 

"didn't", "weren't", 'it', 'theirs', 'our', 'wouldn', 'am', 'was', 'be', 'they', 'on', "you'd", 'out', "haven't", 'hasn', 'that', 'no', 'yourself', 'o', 'into', 'herself', 'has', 'your', 'against', 'shouldn', 'having', 'if', 'had', 'a', 'needn', 'can', 'when', 'under', 'doesn'}
```

<br/>

- 예시 문장에서 stopwords를 제거하는 과정

```python
example_sentence = "This is an example showing off stop word filtration."
words = word_tokenize(example_sentence)
print(words)

# 불용어를 제거하는 필터 만들기
filtered_sentence = list()

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
        
print(filtered_sentence)

# 위와 같은 결과를 출력하는데, 더 간단하게 할수 있는 방법
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)
```

##### 실행결과
```
['This', 'is', 'an', 'example', 'showing', 'off', 'stop', 'word', 'filtration', '.']  


### stopwords 제거 후 ↓  


['This', 'example', 'showing', 'stop', 'word', 'filtration', '.']
```

<br/>

-------

### 3. Stemming words with NLTK
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))
```
##### 실행결과
```
python
python
python
python
pythonli
```

- 문장으로 확장
```python
new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once"

stem = list()
words = word_tokenize(new_text)
for w in words:
    stem.append(ps.stem(w))
print(stem)
```
##### 실행결과
```
['It', 'is', 'veri', 'import', 'to', 'be', 'pythonli', 'while', 'you', 'are', 'python', 'with', 'python', '.', 'All', 'python', 'have', 'python', 'poorli', 'at', 'least', 'onc']
```

<br/>

-------

### 4. Part of Speech Tagging 
- Pos Tagging : 

```python
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
# unsupervised_tokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content(sentences):
    try:
        for i in sentences:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
            
    except Exception as e:
        print(str(e))
        
process_content(tokenized)
```
##### 실행결과
```
[('PRESIDENT', 'NNP'), ('GEORGE', 'NNP'), ('W.', 'NNP'), ('BUSH', 'NNP'), ("'S", 'POS'), ('ADDRESS', 'NNP'), ('BEFORE', 'IN'), ('A', 'NNP'), ('JOINT', 'NNP'), ('SESSION', 'NNP'), ('OF', 'IN'), ('THE', 'NNP'), ('CONGRESS', 'NNP'), ('ON', 'NNP'), ('THE', 'NNP'), ('STATE', 'NNP'), ('OF', 'IN'), ('THE', 'NNP'), ('UNION', 'NNP'), ('January', 'NNP'), ('31', 'CD'), (',', ','), ('2006', 'CD'), ('THE', 'NNP'), ('PRESIDENT', 'NNP'), (':', ':'), ('Thank', 'NNP'), ('you', 'PRP'), ('all', 'DT'), ('.', '.')]
...
```

<br/>

-------

### 5. Chunking with NLTK
- Chunking : grouping과 관련이 있다.
- One of the main goals of chunking is to group into what are known as "noun phrases." The idea is to group nouns with the words that are in relation to them.

```python
def process_content(sentences):
    try:
        for i in sentences:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            
            chunked = chunkParser.parse(tagged)
            
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content(tokenized)
```
##### 실행결과

![zz](https://user-images.githubusercontent.com/37234822/60866752-1dafca80-a264-11e9-9ac0-1ccefaa84c24.JPG)

<br/>

-------

### 6. Chinking with NLTK
- chinking : chunking 한 것에서 빼고싶은 것만 뺄 수 있는 것. 해당되는 품사를 제외시켜주는 역할을 한다. chunking은 {}로 묶어서 나타낼 품사를 정하는 반면, chinking은 }{로 제외시킬 품사를 정한다.

```python
chunkGram = r"""Chunk: {<.*>+}
                        }<VB.?|IN|DT>+{"""
```


<br/>

-------

### 8. Lemmatizing with NLTK
- A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
- The only major thing to note is that lemmatize takes a part of speech parameter, "pos." If not supplied, the default is "noun."
- *n-명사, v-동사, a-형용사, r-부사

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats")) # -> cat
print(lemmatizer.lemmatize("cacti")) # -> cactus
print(lemmatizer.lemmatize("geese")) # -> goose
print(lemmatizer.lemmatize("rocks")) # -> rock
print(lemmatizer.lemmatize("python")) # -> python
print(lemmatizer.lemmatize("better", pos="a")) # -> good
print(lemmatizer.lemmatize("best", pos="a")) # -> best
print(lemmatizer.lemmatize("run")) # -> run
print(lemmatizer.lemmatize("run",'v')) # -> run
```

<br/>

-------

### 10. Wordnet with NLTK
```python
from nltk.corpus import wordnet
syns = wordnet.synsets("program")
print(syns)
```
##### 실행결과
```
[Synset('plan.n.01'), Synset('program.n.02'), Synset('broadcast.n.02'), Synset('platform.n.02'), Synset('program.n.05'),

Synset('course_of_study.n.01'), Synset('program.n.07'), Synset('program.n.08'), Synset('program.v.01'), Synset('program.v.02')]
```

<br/>

- 'program'은 많은 뜻을 가지고 있다. 아래는 다양한 방법
```python
print(syns[0])
# Synset('plan.n.01')

print(syns[0].lemmas())
# [Lemma('plan.n.01.plan'), Lemma('plan.n.01.program'), Lemma('plan.n.01.programme')]

print(syns[0].lemmas()[0].name)
# <bound method Lemma.name of Lemma('plan.n.01.plan')>

print(syns[0].name)
# <bound method Synset.name of Synset('plan.n.01')>

print(syns[0].definition())
# a series of steps to be carried out or goals to be accomplished

print(syns[0].examples())
# ['they drew up a six-step plan', 'they discussed plans for a new bond issue']
```

<br/>
<br/>

- 단어의 동의어, 반의어
```python
synonyms = list()
antonyms = list()

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
```
```python
# 동의어
print(set(synonyms))
```
```
{'well', 'goodness', 'in_force', 'honorable', 'thoroughly', 'skilful', 'trade_good', 'soundly', 'salutary', 'beneficial', 'secure', 'respectable', 'good', 'serious', 'expert', 'ripe', 'in_effect', 'unspoiled', 'adept', 'near', 'honest', 'undecomposed', 'upright', 'right', 'full', 'commodity', 'sound', 'estimable', 'skillful', 'practiced', 'effective', 'safe', 'unspoilt', 'just', 'dependable', 'dear', 'proficient'}
```
```python
# 반의어 
print(set(antonyms))
```
```
{'evilness', 'evil', 'ill', 'badness', 'bad'}
```

<br/>
<br/>

- 단어의 유사도
```python
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2)) # 0.9091
 
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("hat.n.01")

print(w1.wup_similarity(w2)) # 0.5

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

print(w1.wup_similarity(w2)) # 0.6957

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")

print(w1.wup_similarity(w2)) # 0.32
```
