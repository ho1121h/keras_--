1. 토근화

- 단어 토근화
    - 토큰의 기준을 단어로 하는 경우(가장 쉽게는 띄어쓰기 기준으로 분리)
-> 특수 기호, 구두점도 고려해야한다. 돈을 의미하는 $, 구분점 .(192.168.0.1) 등...
줄임말과 단어 띄어쓰기 등등...
- 문장 토근화
    - 토큰의 단위가 문장인 경우
영어의 경우 NLTK, 한국어의 경우 kss가 대표적인 라이브러리 존재.

2. 정제, 정규화
- 정제 : 노이즈 데이터 제거
- 정규화 :표현 방법이 다른 언어들을 통합시켜 같은 단어로 만든다.
    
    1. 규칙에 기반한 단어들 통합
    같은 의미지만 다른 언어로 표기되는 것들.
    2. 대, 소문자 통합
경우에 따라서는 맨 앞단어만 소문자, 다른단어는 대문자로 바꾸거나 전부다 소문자로 하는경우 등 다양하다 (케바케)
    3. 불필요한 단어 제거(불용어나.. 본인판단해서)
    아무 의미도 없는 글자들을 의미하기도 하며, 분석하고자 하는 목적에 맞지 않는 단어들
        - 등장 빈도가 적은 단어
        - 길이가 짧은 단어

3. 정수 인코딩
    - dict 형태를 가져와 빈도수 측정. , 또는 NLTK 의 FreqDist를 사용하여 빈도 수 계산 가능.
```py

dict_name.most_common(size) # 등장 빈도수가 높은 상위 size 단어
```
```py
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences)

```
4. 패딩
기본적으로 모든 문장의길이(정수화된 텍스트)는 다르기 때문에 이를 모두 같은 길이로 만들어 하나의  행렬로 만들어 한번에 처리 가능케 함.
- 제로 패딩
```py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

max_len = max(len(item) for item in encoded)
print('최대 길이 :',max_len)

for sentence in encoded: # 각 문장에 대해서
    while len(sentence) < max_len: # max_len보다 작으면
        sentence.append(0)

padded_np = np.array(encoded)
print(padded_np)

```
- 원핫 인코딩
정수화 기법 중 가장 기본적인 표현 방법. 단어 집합의 크기를 벡터의 차원으로 하고 표현하고 싶은 단어의 인덱스에 1의 값을 부여한다. 참 거짓의 표현
- 카테고리 클래스 사용
```py
from tensorflow.keras.utils import to_categorical

tokenizer = Tokenizer()
text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer.fit_on_texts([text])
tokenizer.word_index
>>
단어 집합 : {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)
>> [2, 5, 1, 6, 3, 7]

one_hot = to_categorical(encoded)
print(one_hot)
>>
[[0. 0. 1. 0. 0. 0. 0. 0.] #인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] #인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] #인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] #인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] #인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
```