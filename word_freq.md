## 글에서 단어의 빈도분포 ('Hamlet.txt' 예시)
<br/>

### 1. 가장 많이 나오는 단어 10개  

```python
from collections import Counter

f = open('Hamlet.txt', 'r')

wordDict = Counter()

sentences = f.readlines()

for sentence in sentences:
    for word in sentence.split():
        wordDict[word] += 1
        
for word, freq in wordDict.most_common(10): # 반복문으로 실행결과 확인하기
    print(word, freq)
   
most_common = wordDict.most_common(10) # 변수명으로 저장하기
most_common
```


##### 실행결과
```
[('the', 929),
 ('and', 680),
 ('of', 625),
 ('to', 608),
 ('I', 523),
 ('a', 453),
 ('my', 444),
 ('in', 382),
 ('you', 361),
 ('Ham.', 358)]
```
<br/>

-------

### 2. 단어의 빈도 분포 그래프 그리기  

```python
import matplotlib.pyplot as plt

names, values = zip(*most_common)
plt.bar(names, values, color='orange')
plt.xlabel('words')
plt.ylabel('frequency')
plt.title('Frequency of words')
```

![freqdist](https://user-images.githubusercontent.com/37234822/60784749-19fc4500-a18b-11e9-938f-004edf18d082.png)

<br/>

-------

### 3. 워드클라우드로 나타내기
```python
from wordcloud import WordCloud

wc = WordCloud(width = 1000, height = 600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(wordDict))
plt.axis("off")
plt.show()
```

![wordCloud](https://user-images.githubusercontent.com/37234822/60784760-254f7080-a18b-11e9-8767-7880bd090f4f.png)

