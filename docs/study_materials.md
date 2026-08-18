Study materials
---


## 1. Podstawowe pojęcia
#### Źródła
- [pubmed](https://pmc.ncbi.nlm.nih.gov/articles/PMC2394262/pdf/89-6601118a.pdf)

*Do czego wykorzysujemy analizę przezywalności?*
W badaniach nad np. nowotowrami interesuje nas przewidywanie czasu 
do jakiegoś zdarzenia *survival time*. Moze być to równiez czas do 
uzyskania remisji, nawrotu choroby, śmierci pacjenta. 

*Czemu mamy specjalne metody do analizy przezywalności?*
Gdyby wydarzenie, do którego czas badamy występowało zawsze, 
zwykłe metody analizy danych nie miałyby z tym problemu.

Ale, w naszych danych wydarzenie nie zawsze musi się odbyć w badanym
okresie czasu. To sprawia, ze w danych występuja dane o wartości *unknown*.

Dane przezywalnosci zadko kiedy maja rozklad normalny. Sa skrzywione,
zazwyczaj mamy więce wydarzeń na pocztku badanego czasu i mało przy jego końcu.

#### Słowniczek

- CENSORING - przez badany czas pacjent nie doświadczył 'zdarzenia'
przyczyny moga być rózne: wydarzyło się później lub w inny sposób zostało wykluczone

1. Right Censoring (The Most Common)
znamy czas poczatku obceswacji
ale pacjent wychodzi lub kończy się czas badania przed danym wydarzeniem
wiemy tylko, ze czas do wydarzenia jest wiekszy niz obserwowany 
przykład: czas badania - 5 miesiecy, wiemy ze: ($T > 5$ mies.)

2. Left Censoring
 'event' juz wydarzyl sie przed rozpoczeciem obserwacji 
 wiemy, ile czasu konkretnie przed obserwacja lub tez nie 
 przykład: przychodzi do nas pacjent po operacji, operacja miała miejsce 
 3 misiace temu. W miedzyczasie pacjent doswiadczyl nawrotu choroby. 
 Wiemy, ze czas ($T <= 3$ mies.). 
 
3. Interval Censoring
Wiemy, ze okreslnone wydzarzenie mialo miejsce pomiedzy dwoma punktami czasowymi.
Co 3 miesiace pacjent przychodzi do kontroli. Miedzy 2., a 3. kontrola pojawił sie
nawrot choroby. Wiemy, ze ($3 < T <=> 6$ mies.)


- SURVIVAL & HAZARD
Jak modelujemy dane?: szacujemy 2 wartości prawdopodobieństwa

1. Survival S(t) - prawdopodobienstow, ze pacjent przetrwa do danego czasu t

2. Hazard or risc h(t) - prawdopodobienstwo, ze pacjent doswiadczy zdarzenia 
'event' w danym czasie t, jest to wiec prawdop. wydarzenia w czase t u pacjenta,
ktory przetrwal do czasu t

- KAPLAN-MEYER SURVIVAL ESTIMATE (KM or product-limit method)

S(tj) = S(t_{j-1}) * (1 - dj / nj)
nj - liczba pacjentow przed zdarzeniem w czasie j
dj - liczba wydarzen w czasie tj

- HAZARD AND CUMULATIVE HAZARD

h(t) = -d/dt[log(S(t))]
- nie ma prostego sposobu na estymowanie h(t)
- aby tego nie robić estymuje sie H(t) - czyli ryzyko skumulowane

H(t) = -log(S(t))
- trudna jest interpretacja H(t)
- mozna myslec o H(t) jak o liczbie wydarzen ktore maja miejsce do czasu t 


## 2. Funkcja przezywalnosci (Survival function)
#### Źródła
- [wiki](https://en.wikipedia.org/wiki/Survival_function)

S(t) = P(T > t)

- T - cumulative and continuous random variable describing the time to failure 



