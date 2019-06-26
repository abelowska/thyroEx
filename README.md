# Automatyczna klasyfikacja binarna guzów tarczycy

## 1. Cel i opis projektu

Obecnie, w przypadku wykrycia guza tarczycy badaniem usg, najczęściej wykonywanym badaniem pozwalającym zadecydować o 
stopniu złośliwości guza jest biopsja tarczycy (BACC). Niestety, guzy tarczycy należą do najtrudniej klasyfikowalnych zmian,
 z powodu wielu cech na podstawie których jest dokonywana ocena złośliwości.
 Ponadto cechy te nie sa określone w sposób jasny i powtarzalny, dlatego występuje tu duży
 subiektywizm lekarza podejmującego decyzję.
 
 
 Pomocnym narzedziem dla lekarzy stał się  system opisu guzów tarczycy TI-RADS
 (Thyroid Imaging Reporting and Data System), który ma na celu unifikacje języka
 opisu zmian tarczycy. Poniżej, na obrazku przedstawiono sposób klasyfikacji.
 
 ![alt text][logo]

[logo]: https://github.com/abelowska/IM-Detector/master/dataTIRADS-2017-Flow-Chart.jpg
 "Logo Title Text 2"
 
Celem naszego projektu była automatyczn klasyfikacja guzów tarczycy na złośliwe i niezłośliwe. 
Do dokonania klasyfikacji została użyta pre trenowana sieć neuronowa GoogLeNet. Dla podziału guzów na złośliwe i nie 


https://github.com/dandxy89/ImageModels/blob/master/GoogLeNet.py

https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14