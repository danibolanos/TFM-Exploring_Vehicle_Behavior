## TFG - Tools to Ensure Fairness in Machine Learning

Repositorio de mi **trabajo de fin de grado** para el *Doble Grado en Ingeniería Informática y Matemáticas* de la [Universidad de Granada](http://www.ugr.es) sobre un estudio experimental para diferentes modelos de equidad contrafactual. Puede descargar una versión compilada de la memoria en [este enlace](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/releases/download/v1.0.0/TFG.Herramientas_para_Garantizar_Justicia_en_Aprendizaje_Automatico.pdf). Puede descargar las diapositivas de la presentación realizada al tribunal [aquí](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/releases/download/v1.0.0/TFG.Presentacion.pdf) (plantilla cedida por Julio Mulero de la Universidad de Alicante).

Un tutorial para la ejecución del experimento basado en *Jupyter Notebook* puede ser consultado en el [siguiente enlace](https://github.com/danibolanos/TFG-Guarantee_Fairness_in_ML/blob/main/experimentos/tutorial.ipynb). Si desea ejecutarlos en su ordenador, será necesario que descargue las dependencias a los siguientes paquetes:

* Pandas 1.2.4
* NumPy 1.19.2
* Scikit-learn 0.24.2
* matplotlib 3.4.3
* seaborn 0.11.2
* pathlib2 2.3.6
* Aequitas 0.42.0
* PyStan 2.19.1.1
* [PyMC3](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)) (opcional; para *otras_pruebas.py*)

Puede hacerlo, bien manualmente, o bien puede utilizar el siguiente comando:

```
python -m pip install -r requirements.txt
 ```
 
#### Mejoras en el rendimiento

En un futuro, se propone mejorar los resultados predichos para el modelo de regresión, utilizando un modelo no lineal que se adapte a las distribuciones estudiadas. Por ejemplo, utilizando un modelo de regresión polinomial o segmentada, o redefiniendo el problema en el ámbito de la clasificación y utilizando el algoritmo SVM o una red neuronal para predecir los resultados de las etiquetas.
 
### Description

It proposes as a challenge, after an exhaustive study of the existing bibliography and tools on the subject, to carry out a comparative analysis in *Python* that includes fairness evaluations of all the families and visualisations for the interpretation of results and models.

To do so, we will start by carrying out an exhaustive review of the different mathematical formalisations of the concept of fairness in the literature, analysing the advantages and disadvantages of each one both from a theoretical point of view and in terms of their practical implementation in specific classification problems by means of automatic learning. In addition, a study and classification of the most popular published algorithms that work with fairness will be carried out and a study of the advantages and disadvantages of each one will be made.

-----

### Main Bibliography

- Barocas, S., Hardt, M., & Narayanan, A. (2018). Fairness and Machine Learning. fairmlbook. org, 2018. URL: http://www.fairmlbook.org.

- Mitchell, S., Potash, E., Barocas, S., D'Amour, A., & Lum, K. (2018). Prediction-based decisions and fairness: A catalogue of choices, assumptions, and definitions. arXiv:1811.07867.

- Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).

- M. J. Kusner, J. R. Loftus, C. Russell & R. Silva (2018). Counterfactual Fairness.

- Saleiro, P., Kuester, B., Hinkson, L., London, J., Stevens, A., Anisfeld, A., ... & Ghani, R. (2018). Aequitas: A bias and fairness audit toolkit. arXiv:1811.05577.
