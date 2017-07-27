# 해당 문서는

`Word2Vec` 와 `Doc2Vec`을 공부하면서 만든 ipynb 파일입니다.

[시작할 때 도움되는 페이지](https://www.lucypark.kr/courses/2015-dm/text-mining.html)

## 요구되는

[Tensorflow](https://www.tensorflow.org/) An open-source software library for Machine Intelligence  
```pip install tensorflow-gpu```   
[NLTK](http://www.nltk.org/): Provides modules for text analysis (mostly language independent)  
```pip install -U nltk```   
[KoNLPy](http://konlpy.org/en/v0.4.4/): Provides modules for Korean text analysis  
```pip install konlpy```   
[Gensim](https://radimrehurek.com/gensim/): Provides modules for topic modeling and calculating similarities among documents  
```pip install -U gensim```   
[Twython](https://github.com/ryanmcgrath/twython): Provides easy access to Twitter API  
```pip install twython```   


---

## 네이버 기사 분류 

* Train data : 16만개
* Test data : 약 3만3천개
* Class : 48개

|정치      |경제        |사회      |생활/문화     |세계          |IT/과학       |
|:---------|:-----------|:---------|:-------------|:-------------|:-------------|
|청와대    |금융        |사건사고  |건강정보      |아시아/호주   |모바일        |
|국회/정당 |증권        |교육      |자동차/시승기 |미국/중남미   |인터넷/SNS    |
|북한      |산업/재계   |노동      |도로/교통     |유럽          |통신/뉴미디어 |
|행정      |중기/벤처   |언론      |여행/레저     |중동/아프리카 |IT 일반       |
|국방/외교 |부동산      |환경      |음식/맛집     |세계 일반     |보안/해킹     |
|정치일반  |글로벌 경제 |인권/복지 |패션/뷰티     |NA            |컴퓨터        |
|NA        |생활경제    |식품/의료 |공연/전시     |NA            |게임/리뷰     |
|NA        |경제 일반   |지역      |책            |NA            |과학 일반     |
|NA        |NA          |인물      |종교          |NA            |NA            |
|NA        |NA          |사회일반  |날씨          |NA            |NA            |
|NA        |NA          |NA        |생활문화 일반 |NA            |NA            |

---

## 모형별  Accuracy
각 모형별 notebook 파일은 세부폴더 안에 있음


No.| NLP Model|	Part-of-speech|	Window|	Train Accuracy|	Test Accuracy|	Top2 Accuracy
-------|-----------|-------------|----------------|-----------|--------------|----------------------
1|	PV-DBOW|	Alldel(punc., foreign)|	10|	0.81343|	0.66084|	0.78832
2|	PV-DBOW|	Alldel(punc., foreign)|	20|	0.81263|	0.65843|	0.78733
3|	PV-DM|	Alldel(punc., foreign)|	10|	0.64322|	0.47035|	0.61242
4|	PV-DM|	Alldel(punc., foreign)|	20|	0.57794|	0.40563|	0.53979
5|	PV-DBOW|	Noun + Verb|	10|	0.80410|	0.64285|	0.77354
6|	PV-DBOW|	Noun + Verb|	20|	0.80196|	0.64041|	0.77076
7|	PV-DM|	Noun + Verb|	10|	0.65112|	0.46636|	0.60704
8|	PV-DM|	Noun + Verb|	20|	0.58591|	0.40007|	0.52783
9|	DBOW + DM|	|1+3|		0.87441|	0.64235|	0.76603
10|	DBOW + DM|	|5+7|		0.86920|	0.62822|	0.75233


### 예시

No.1 모형 각 class별 acc 출력도


Accuracy            | Label
:-------------------|:---------------------
0.35166|IT/과학-IT 일반
0.68814|IT/과학-게임/리뷰
0.64222|IT/과학-과학 일반
0.62952|IT/과학-모바일
0.88826|IT/과학-보안/해킹
0.63194|IT/과학-인터넷/SNS
0.69289|IT/과학-컴퓨터
0.72874|IT/과학-통신/뉴미디어
0.44539|경제-경제 일반
0.75067|경제-글로벌 경제
0.68758|경제-금융
0.84412|경제-부동산
0.37103|경제-산업/재계
0.55144|경제-생활경제
0.68750|경제-중기/벤처
0.70628|경제-증권
0.61277|사회-교육
0.68095|사회-노동
0.72977|사회-사건사고
0.37713|사회-사회일반
0.46299|사회-식품/의료
0.62602|사회-언론
0.62142|사회-인권/복지
0.65912|사회-인물
0.47227|사회-지역
0.69475|사회-환경
0.72897|생활/문화-건강정보
0.79708|생활/문화-공연/전시
0.88095|생활/문화-날씨
0.72543|생활/문화-도로/교통
0.37327|생활/문화-생활문화 일반
0.71172|생활/문화-여행/레저
0.57091|생활/문화-음식/맛집
0.83777|생활/문화-자동차/시승기
0.90574|생활/문화-종교
0.83862|생활/문화-책
0.81295|생활/문화-패션/뷰티
0.63385|세계-미국/중남미
0.77608|세계-세계 일반
0.64448|세계-아시아/호주
0.70796|세계-유럽
0.79229|세계-중동/아프리카
0.58520|정치-국방/외교
0.70053|정치-국회/정당
0.71275|정치-북한
0.49057|정치-정치일반
0.68759|정치-청와대
0.46246|정치-행정


