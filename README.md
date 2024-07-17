# Word2Vec implementation with Python and Cython 

Model trained on 100mb Text8 corpus and evaluated on [questions-words.txt](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)

Average training speed:
```
+------------------------------+------------------+
|            method            | words per second |
+------------------------------+------------------+
|         Only Cython          |      12849       |
|    Cython + Sigmoid Table    |      13328       |
|        Cython + BLAS         |      26559       |
| Cython + BLAS+ Sigmoid Table |      28746       |
+------------------------------+------------------+
```

Accuracy:
![Accuracy](https://github.com/pkhanzhina/Word2vec/blob/main/plots/accuracy_plot_total.png)
Accuracy per section:
![Accuracy per section](https://github.com/pkhanzhina/Word2vec/blob/main/plots/accuracy_plot_by_sections.png)
