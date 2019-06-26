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
 
 ![alt text][tirads]

[tirads]: https://github.com/abelowska/IM-Detector/blob/master/src/images/TIRADS-2017-Flow-Chart.jpg

 
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


 ![alt text][flow]

[flow]: https://github.com/abelowska/IM-Detector/blob/master/src/images/flow.png


* Dla poprawnego działania sieci potrzebna była normalizacja danych. Ponieważ wszystkie pliki były w formacie .jpg,
nie było wiadomo na jakim przybliżeniu było wykonane zdjęcie USG. W tym celu jest wykrywana
ilość pikseli pomiędzy punktami 0 oraz 1 zaznaczonymi na osiach skali obrazów.

* Aby zminimalizowac liczbę zakłóceń na obrazach, które moga wpłynąć na działanie sieci, 
artefakty obecna na obrazach, powstałe w trakcie pracy lekarza, takie jak znaczniki czy linie pomiarowe,
oraz wyżej wspomniane osie, są wykrywane i wyczerniane. Natępnie, metodą najbliższych sąsiadów obraz 
jest odbudowywany. Odbudowanie nie jest idealne, jednak zaciera się całkowicie w  trakcie zmniejszania obrazów.

* Przy analizie danych medycznych nie poleca się robienia klasycznej augmentacji danych, poniewaz może to znacznie pogorszyć
wyniki. Jednak, aby uzyskac choćby minimalną skuteczność sieci, potrzebowaliśmy znacznie więcej danych.
Zdecydowaliśmy się na augmentacje danych polagającą na nieznacznym rotowaniu obrazu (od -10 do 10 stopni co 1 stopień).
Uzyskaliśmy w ten sposób zbiór danych liczący 7302. próbki.


### 3.2 Trenowanie sieci neuronowej

Naszym modelem była siec neuronowa GoogLeNet. Posiada ona 96 warstw. Aby skorzystać z pre-trenowanego modelu
użyliśmy pliku z gotowymi wagami dla tej architektury.

```python
# Create model
input, model = create_googlenet("../googLeNet/googlenet_weights.h5")
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

```

Nasz zbiór danych został podzielony w stosuku 25 : 75 na testowy i treningowy. Ilość epok była równa 15, a ``batch_size`` wynosił 32.

Aby trenować tylko górne warstwy sieci wyłączyliśmy aktualizowanie wag modelu dla niektórych warstw. Następnie, aby
móc wykorzystać gotowy model musieliśmy podmienić wastwy końcowe, które oryginalnie miały 1000 klas wynikowych.
W tym celu wpięliśmy utworzone warstwy do modelu:

```python
# Plugging into old net
loss1_classifier = Dense(2, name='loss1/classifier', kernel_regularizer=l2(0.0002))(model.layers[100].output)
loss2_classifier = Dense(2, name='loss2/classifier', kernel_regularizer=l2(0.0002))(model.layers[101].output)
loss3_classifier = Dense(2, name='loss3/classifier', kernel_regularizer=l2(0.0002))(model.layers[102].output)

loss1_classifier_act = Activation('softmax')(loss1_classifier)
loss2_classifier_act = Activation('softmax')(loss2_classifier)
loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)
```

Oryginalnie GoogLeNet posiada trójwymiarowy wektor wyjścia:

``googlenet = Model(input=input, output=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])``

aby wsteczna propagacja informacji docierała do warstw głębokich, ponieważ jest to bardzo głęboka sieć.
Zdecydowaliśmy się jednak, ze względu na trenowanie tylko górnych warstw, na zmianę wektora wyjściowego na jednowymiarowy:

``
googlenet = Model(inputs=model.layers[0].output,
                  outputs=[loss3_classifier_act])
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)``


Dzięki temu zeszlismy w ilości trenowalnych parametrów do 5,975,602.


## 4. Stos technologiczny

* Python 3.6
* Keras 2.2.4
* Theano 1.0.4
* googLeNet
* OpenCv
* sklearn
* Pillow
* imageio
* piexif

## 5. Wyniki

Przy obecnej implementacji nie uzyskalismy satysfakcjonujących wyników. 
Koncowe parametry naszego modelu wynoszą:


``loss: 4.3033 - acc: 0.7941 - val_loss: 4.5435 - val_acc: 0.785
``

Jednak przeprowadzony test nie wykorzystywał w pełni możliwości stworzonego modelu. 
Nie udało nam się uruchomić modelu dla całego poisadanego zbioru danych.

Zauważylismy również, iż do ostatniej epoko wartość zmiennej ``loss`` ciągle malała, co sugeruje, iż możemy dalej trenowac nasz model z większa ilością epok.

## 5. Perspektywy rozwoju projektu

Jak to przy sieciach neuronowych bywa, jest wiele obszarów, które musimy sprawdzić, aby poprawić skuteczność naszej sieci. Nasz plan rozwoju projektu zawiera:

* obserwację wpływu zamrażania niektórych warstw sieci na jej skuteczność
* zastosowanie trójwymiarowego wektora wyjścia dla modelu
* zwiększenie ilości epok
* zmiana wielkości `` batch_size``
* zastosowanie dodatkowych filtrów dla posiadanego zbioru danych


## 6. Bibliografia

https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14