# Automatyczna klasyfikacja binarna guzów tarczycy

## 1. Cel i opis projektu

Obecnie, w przypadku wykrycia guza tarczycy badaniem usg, najczęściej wykonywanym badaniem pozwalającym zadecydować o 
stopniu złośliwości guza jest biopsja tarczycy (BACC). Niestety, guzy tarczycy należą do najtrudniej klasyfikowalnych zmian,
 z powodu wielu cech, na podstawie których jest dokonywana ocena złośliwości.
 Ponadto, cechy te nie sa określone w sposób jasny, dlatego opis jest obciążony dużym
 subiektywizmem lekarza.
 
 
 Pomocnym narzedziem dla lekarzy stał się  system opisu guzów tarczycy TI-RADS
 (Thyroid Imaging Reporting and Data System), który ma na celu unifikacje języka
 opisu zmian tarczycy. Poniżej, na obrazku przedstawiono sposób klasyfikacji.
 
 ![alt text][logo]

[logo]: https://github.com/abelowska/IM-Detector/blob/master/src/images/TIRADS-2017-Flow-Chart.jpg

 
 Na postawie ilości przyznanych punktów okresla się przynależność guza do danej kategorii. Klasy 1-3 traktowane sa jako
 zmiany niezłośliwe, wymagające obserwacji, klasy 4-5 opisują złośliwe nowotwory tarczycy. 
 Dzięki użyciu TI-RADS-u można by było ograniczyć ilość wykonywanych biopsji, a co za tym idzie, zmiejszyć koszty związane
 z przeprowadzeniem tego badania. Jednak, ze względu na trudność określenia niektórych cech, takich jak na przykład nieregularność obrysu guza,
 problem nie został rozwiązany. 

Celem naszego projektu było stworzenie automatycznego klasyfikatora guzów tarczycy na złośliwe i niezłośliwe przy użyciu sieci neuronowej.
Do dokonania klasyfikacji została użyta pre trenowana sieć neuronowa GoogLeNet. Nasze dane zostały opisane wg. rozszerzonej skali TI-RADS (z użyciem subklas
4a, 4b oraz 4c) a nastepnie podzielone na dwie klasy dla naszego binarnego modelu: złośliwe i niezłośliwe.


## 2. Przeglad istaniejących rozwiazań

## 3. Opis implementacji

Projekt składał się z dwóch głównych cześci: pre-processingu danych oraz z trenowania modelu.

### 3.1 Pre-processing danych

Zbiór danych na którym pracowaliśmy zawiera 355 zdjęć: 63 przypadków niezłośliwych oraz 292 przypadków złośliwych.
Wszystkie dane zostały pobrane z bazy danych kolumbijskiego uniwersytetu medycznego o wolnym dostepie.

Obróbkę zdjęć przedstawia poniższy diagram:


 ![alt text][logo]

[logo]: https://github.com/abelowska/IM-Detector/blob/master/src/images/flow.png

Dla poprawnego działania sieci potrzebna była normalizacja danych. Ponieważ wszystkie pliki były w formacie .jpg,
nie było wiadomo na jakim przybliżeniu było wykonane zdjęcie USG. W tym celu jest wykrywana
ilość pikseli pomiędzy punktami 0 oraz 1 zaznaczonymi na osiach skali obrazów.

Aby zminimalizowac liczbę zakłóceń na obrazach, które moga wpłynąć na działanie sieci, 
artefakty obecna na obrazach, powstałe w trakcie pracy lekarza, takie jak znaczniki czy linie pomiarowe,
oraz wyżej wspomniane osie, są wykrywane i wyczerniane. Natępnie, metodą najbliższych sąsiadów obraz 
jest odbudowywany. Odbudowanie nie jest idealne, jednak zaciera się całkowicie w  trakcie zmniejszania obrazów.

Przy analizie danych medycznych nie poleca się robienia klasycznej augmentacji danych, poniewaz może to znacznie pogorszyć
wyniki. Jednak, aby uzyskac choćby minimalną skuteczność sieci, potrzebowaliśmy znacznie więcej danych.
Zdecydowaliśmy się na augmentacje danych polagającą na nieznacznym rotowaniu obrazu (od -10 do 10 stopni co 1 stopień).
Uzyskaliśmy w ten sposób zbiór danych liczący 7302. próbki.


### 3.2 Trenowanie sieci neuronowej




## 4. Stos technologiczny

## 5. Wyniki





https://github.com/dandxy89/ImageModels/blob/master/GoogLeNet.py

https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14