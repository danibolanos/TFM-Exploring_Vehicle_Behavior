"""
Replicación en Python del caso de estudio del artículo Counterfactual Fairness (Kusner et al. 2017)
Y análisis de los modelos con la herramienta software Aequitas para el estudio de equidad
sobre problemas basados en conjuntos de datos del mundo real.

Autor: Daniel Bolaños Martínez
"""

import os
import shutil
import pystan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from pathlib import Path
import seaborn as sns
import copy
import matplotlib.pyplot as plt

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

def crear_borrar_directorio(directorio, flag):
    if flag:
        try:
            os.mkdir(directorio)
        except OSError:
            print("La creación del directorio %s falló" % directorio)
    else:
        try:
            shutil.rmtree(directorio)
        except OSError:
            print("La eliminación del directorio %s falló" % directorio)
        

# Crea un diccionario compatible con pystan para el conjunto de datos y los atributos protegidos dados
def create_pystan_dic(data, protected_attr, train=True):
    dic_data = {}
    dic_data['N'] = len(data)
    dic_data['C'] = len(protected_attr)
    dic_data['A'] = np.array(data[protected_attr])
    dic_data['GPA'] = list(data['UGPA'])
    dic_data['LSAT'] = list(data['LSAT'])
    if train:
        dic_data['FYA'] = list(data['ZFYA'])
    
    return dic_data

# Preprocesamiento del conjunto de datos original 'law_data.csv'
def preprocess_data_original():
    # Leemos el conjunto de datos con el nombre pasado por parametro
    dataset = pd.read_csv('./datos/law_data.csv', index_col=0)  
    
    # Creamos una columna que indique con 0 o 1 la pertenencia al sexo Masculino o Femenino
    dataset['sex'] = dataset['sex'].apply(lambda a: 'Male' if a == 2 else 'Female')
    
    dataset['entity_id']= np.arange(1,len(dataset.iloc[:,0])+1,1)
    
    dataset['label_value'] = dataset['first_pf'].apply(lambda a: 0 if a == 0.0 else 1)
    
    dataset['race'] = dataset['race'].apply(lambda a: 'Hispanic' if a == 'Mexican' else a)
    dataset['race'] = dataset['race'].apply(lambda a: 'Hispanic' if a == 'Puertorican' else a)
    dataset['race'] = dataset['race'].apply(lambda a: 'Other' if a == 'Amerindian' else a)
    
    # Utilizamos el criterio negativo o 0 a 0.0 y positivo a 1.0 para la puntuacion
    dataset['score'] = dataset['ZFYA'].apply(lambda a: 0.0 if a <= 0 else 1.0)
    
    # Eliminamos las columnas no necesarias
    dataset = dataset.drop(['ZFYA'], axis=1)
    dataset = dataset.drop(['LSAT'], axis=1)
    dataset = dataset.drop(['UGPA'], axis=1)
    dataset = dataset.drop(['sander_index'], axis=1)
    dataset = dataset.drop(['region_first'], axis=1)
    dataset = dataset.drop(['first_pf'], axis=1)
    
    #dataset.to_csv("./datos/law_for_aequitas.csv", index=False)

    return dataset


# Preprocesamiento del conjunto de datos 'law_data.csv'
def preprocess_data(data, protected_attr):    

    data['race'] = data['race'].apply(lambda a: 'Hispanic' if a == 'Mexican' else a)
    data['race'] = data['race'].apply(lambda a: 'Hispanic' if a == 'Puertorican' else a)
    data['race'] = data['race'].apply(lambda a: 'Other' if a == 'Amerindian' else a)
    data['first_pf'] = data['first_pf'].apply(lambda a: 0 if a == 0.0 else 1)
    
    # Convertimos la columna 'LSAT' a tipo entero
    data['LSAT'] = data['LSAT'].apply(lambda a: int(a))
    
    # Creamos una columna que indique con 0 o 1 la pertenencia al sexo Masculino o Femenino
    data['Female'] = data['sex'].apply(lambda a: 1 if a == 1 else 0)
    data['Male'] = data['sex'].apply(lambda a: 1 if a == 2 else 0)
    
    # Creamos una columna que indique con 0 o 1 la pertenencia al sexo Masculino o Femenino
    data['sex'] = data['sex'].apply(lambda a: 'Male' if a == 2 else 'Female')

    # Realizamos una división 80-20 de los conjuntos de entrenamiento y test
    train, test = train_test_split(data, random_state = 76592621, train_size = 0.8);
    
    train_orig = copy.deepcopy(train)
    test_orig  = copy.deepcopy(test)
    
    # Separamos cada atributo de raza en una columna con 1 si el individuo 
    # pertenece a ella o 0 si el individuo no pertenece
    train = pd.get_dummies(train, columns=['race'], prefix='', prefix_sep='')
    test = pd.get_dummies(test, columns=['race'], prefix='', prefix_sep='')
    train = train.drop(['sex'], axis=1)  
    test = test.drop(['sex'], axis=1)  
    
    # Creamos un diccionario compatible con pystan para los conjuntos creados anteriormente
    dic_train = create_pystan_dic(train, protected_attr)
    dic_test = create_pystan_dic(test, protected_attr)

    return dic_train, dic_test, train_orig, test_orig
    
# Modelo Completo: usa todos los atributos para la predicción
def mod_full(dic_train, dic_test):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((dic_train['A'], np.array(dic_train['GPA']).reshape(-1,1), 
                         np.array(dic_train['LSAT']).reshape(-1,1)))
    x_test = np.hstack((dic_test['A'], np.array(dic_test['GPA']).reshape(-1,1), 
                        np.array(dic_test['LSAT']).reshape(-1,1)))
    y = dic_train['FYA']

    # Entrenamiento del modelo sobre el conjunto x_train
    lr_full = LinearRegression()
    lr_full.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds_test = lr_full.predict(x_test)
    
    # Predicción de las etiquetas sobre el conjunto x_train
    preds_train = lr_full.predict(x_train)
    
    # Cálculo del score sobre el conjunto x_test
    score = lr_full.score(x_test, dic_test['FYA'])

    return preds_test, preds_train, score

# Modelo equidad por desconocimiento: no usa los atributos sensibles en predicción
def mod_unaware(dic_train, dic_test):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((np.array(dic_train['GPA']).reshape(-1,1), 
                         np.array(dic_train['LSAT']).reshape(-1,1)))
    x_test = np.hstack((np.array(dic_test['GPA']).reshape(-1,1), 
                        np.array(dic_test['LSAT']).reshape(-1,1)))
    y = dic_train['FYA']
    
    # Entrenamiento del modelo sobre el conjunto x_train
    lr_unaware = LinearRegression()
    lr_unaware.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds_test = lr_unaware.predict(x_test)
    
    # Predicción de las etiquetas sobre el conjunto x_train
    preds_train = lr_unaware.predict(x_train)
    
    # Cálculo del score sobre el conjunto x_test
    score = lr_unaware.score(x_test, dic_test['FYA'])

    return preds_test, preds_train, score

# Creamos un diccionario con la media de los parámetros útiles para el modelo de 'K'
# obtenidos en el entrenamiento previo para el modelo base
def get_mean_params(samples, dic):
    dic_data = {}
    # Añadimos los parámetros comunes que comparte con el diccionario original    
    param_base = ['N', 'C', 'A', 'GPA', 'LSAT']
    for param in param_base:
        dic_data[param] = dic[param]
        
    # Guardamos la media del vector de valores para los parámetros que utiliza el modelo '*only_k.stan'
    for param in samples.keys():
        if param not in ['K', 'wK_F', 'wA_F', 'sigma2_G', 'lp__']:
            dic_data[param] = np.mean(samples[param], axis=0)

    return dic_data

# Entrenamos el modelo para el diccionario dado
def MCMC(dic_post, path_model, path_stan):
    model_fit = Path(path_model)
    
    # Comprobamos si ya existe un archivo con el modelo entrenado
    if model_fit.is_file():
        file = open(path_model, "rb")
        fit_samples = pickle.load(file)
    else:
        # Obtiene muestras desde cero a partir del modelo pasado como parámetro
        model = pystan.StanModel(file = path_stan)
        fit_data = model.sampling(data = dic_post, seed=76592621, chains=1, iter=2000)
        fit_samples = fit_data.extract()
        # Guardamos el modelo entrenado
        file = open(path_model, "wb")
        pickle.dump(fit_samples, file, protocol=-1)
    
    return fit_samples

def fair_learning(dic_train, dic_test):
    modelos_dir = Path("./datos/modelos")
    
    # Comprobamos si ya existe un archivo con el modelo entrenado
    if not modelos_dir.exists():
        crear_borrar_directorio(modelos_dir, True)
        
    # Entrenamos el modelo utilizando el MCMC y obtenemos las muestras para cada punto
    train_samples = MCMC(dic_train, "./datos/modelos/train_k_model.pkl", "./datos/stan/law_school_train.stan")

    # Obtenemos la media de la variable K para train
    train_k = np.mean(train_samples["K"], axis=0).reshape(-1, 1)
    
    # Usamos la distribución de K aprendida y hacemos las medias del resto de variables para test
    dic_means = get_mean_params(train_samples, dic_test)
    
    # Volvemos a inferir sobre el modelo esta vez usando el modelo sin FYA para test
    test_samples = MCMC(dic_means, "./datos/modelos/test_k_model.pkl", "./datos/stan/law_school_only_k.stan")
    # Obtenemos la media de la variable K para test
    test_k = np.mean(test_samples["K"], axis=0).reshape(-1, 1)
    
    return train_k, test_k


# Modelo no determinista: suponemos variable de ruido 'K' para generar la distribución del resto
def mod_fair_k(dic_train, dic_test):
    train_k, test_k = fair_learning(dic_train, dic_test)
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = train_k
    x_test = test_k
    y = dic_train['FYA']
    
    # Entrenamiento del modelo sobre el conjunto x_train
    lr_fair_k = LinearRegression()
    lr_fair_k.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds_test = lr_fair_k.predict(x_test)

    # Predicción de las etiquetas sobre el conjunto x_train
    preds_train = lr_fair_k.predict(x_train)
    
    # Cálculo del score sobre el conjunto x_test
    score = lr_fair_k.score(x_test, dic_test['FYA'])

    return preds_test, preds_train, score

# Estima el error entrenando el modelo sobre el conjunto total de datos para una variable 
# observada pasada por parámetro utilizando los atributos protegidos dados por 'A'
def calculate_eps(dic_train, dic_test, var):
    # Reconstruimos el conjunto total para las variables que vamos a usar
    data_a = np.vstack((dic_train['A'], dic_test['A']))
    data_var = dic_train[var] + dic_test[var]
    
    # Entrenamos un modelo para estimar el error para el parámetro var
    lr_eps = LinearRegression()
    lr_eps.fit(data_a, data_var)
    
    # Calculamos los residuos de cada modelo como eps_var = var - Ŷ_var(A)
    eps_train = dic_train[var] - lr_eps.predict(dic_train['A'])
    eps_test = dic_test[var] - lr_eps.predict(dic_test['A'])
    
    return eps_train, eps_test

# Calcula los histogramas para los errores de cada variable calculada
def graph_eps(eps_train_G, eps_train_L, eps_test_G, eps_test_L):
    eps_G = np.vstack((eps_train_G.reshape(-1,1), eps_test_G.reshape(-1,1)))
    eps_L = np.vstack((eps_train_L.reshape(-1,1), eps_test_L.reshape(-1,1)))
    plt.subplot(1, 2, 1)
    plt.hist(eps_G, color="red", bins=100)
    plt.title("$\epsilon_{GPA}$")
    plt.xlabel("$\epsilon_{GPA}$")
    plt.subplot(1, 2, 2)
    plt.hist(eps_L, color="green", bins=100)
    plt.title("$\epsilon_{LSAT}$")
    plt.xlabel("$\epsilon_{LSAT}$")
    plt.savefig('./figures/epsilons.png')
    plt.show()

# Modelo determinista: añadimos términos de error aditivos independientes de los atributos protegidos
def mod_fair_add(dic_train, dic_test):
    # Estimamos el error para GPA
    eps_train_G, eps_test_G = calculate_eps(dic_train, dic_test, 'GPA')
    # Estimamos el error para LSAT
    eps_train_L, eps_test_L = calculate_eps(dic_train, dic_test, 'LSAT')
    
    #graph_eps(eps_train_G, eps_train_L, eps_test_G, eps_test_L)
    
    x_train = np.hstack((eps_train_G.reshape(-1,1), eps_train_L.reshape(-1,1)))
    x_test = np.hstack((eps_test_G.reshape(-1,1), eps_test_L.reshape(-1,1)))
    y = dic_train['FYA']

    # Entrenamiento del modelo usando los errores de train
    lr_fair_add =  LinearRegression()
    lr_fair_add.fit(x_train, y)

    # Predicción de las etiquetas usando los errores para test
    preds_test = lr_fair_add.predict(x_test)
    
    # Predicción de las etiquetas usando los errores para test
    preds_train = lr_fair_add.predict(x_train)
    
    # Cálculo del score sobre el conjunto x_test
    score = lr_fair_add.score(x_test, dic_test['FYA'])

    return preds_test, preds_train, score

# Obtiene a partir de las predicciones de un modelo un dataFrame con la estructura de Aequitas
def get_aequitas_data(train_orig, preds_train, test_orig, preds_test):
    # Realizamos una copia de los conjuntos de entrenamiento
    train = copy.deepcopy(train_orig)
    test  = copy.deepcopy(test_orig)
    # Creamos la columna de score a partir de las predicciones
    train['score'] = preds_train
    test['score'] = preds_test
    # Concatenamos ambos subconjuntos en un conjunto total
    dataset = pd.concat([train,test])
    dataset.rename(columns={'first_pf':'label_value'}, inplace=True)
    # Utilizamos el criterio negativo o 0 a 0.0 y positivo a 1.0 para la puntuacion
    dataset['score'] = dataset['score'].apply(lambda a: 0.0 if a <= 0 else 1.0)
    # Creamos la columna entity_id
    dataset['entity_id']= np.arange(1,len(dataset.iloc[:,0])+1,1)
    # Eliminamos las columnas que no vayamos a usar
    dataset = dataset.drop(['ZFYA'], axis=1)
    dataset = dataset.drop(['LSAT'], axis=1)
    dataset = dataset.drop(['UGPA'], axis=1)
    dataset = dataset.drop(['sander_index'], axis=1)
    dataset = dataset.drop(['region_first'], axis=1)
    dataset = dataset.drop(['Male'], axis=1)
    dataset = dataset.drop(['Female'], axis=1)
    
    return dataset

# Guarda un gráfico para los atributos raza y sexo dado un dataframe
def save_graficas(data, name, aq_palette, attr):
    sns.countplot(x="race", hue=attr, data=data[dataset_full.race.isin(['Hispanic', 'White', 'Black'])], palette=aq_palette)
    plt.savefig('./figures/model_contrast/'+name+'_race_law.png')
    plt.clf()
    sns.countplot(x="sex", hue=attr, data=data, palette=aq_palette)
    plt.savefig('./figures/model_contrast/'+name+'_sex_law.png')
    plt.clf()

# Devuelve la tabla de las métricas de grupo para el conjunto pasado por parámetro
def tabla_metrica_grupo(data):
    g = Group()
    xtab, _ = g.get_crosstabs(data)
    absolute_metrics = g.list_absolute_metrics(xtab)
    tabla_grupo = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
    return tabla_grupo

# Devuelve la tabla de las métricas de sesgo para el conjunto pasado por parámetro
# También podemos especificar los atributos de referencia
def tabla_metrica_sesgo(data, attr_ref):
    # Calculamos las métricas de grupo
    g = Group()
    xtab, _ = g.get_crosstabs(data)
    # Calculamos las metricas de sesgo
    b = Bias()
    # Establecemos los atributos de referencia
    bdf = b.get_disparity_predefined_groups(xtab, original_df=data, ref_groups_dict=attr_ref, alpha=0.05, mask_significance=True)
    calculated_disparities = b.list_disparities(bdf)
    disparity_significance = b.list_significance(bdf)
    tabla_sesgo = bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]
    return tabla_sesgo

# Devuelve una tabla con si se sumplen o no las medidas de equidad para un cierto umbral
def tabla_medidas_equidad(data, attr_ref, tau=0.8):
    # Calculamos las métricas de grupo
    g = Group()
    xtab, _ = g.get_crosstabs(data)
    # Calculamos las metricas de sesgo
    b = Bias()
    # Establecemos los atributos de referencia
    bdf = b.get_disparity_predefined_groups(xtab, original_df=data, ref_groups_dict=attr_ref, alpha=0.05, mask_significance=True)
    # Definimos las medidas de equidad a partir de la tabla de metricas de sesgo
    f = Fairness()
    # Establecemos el valor del umbral con la variable tau
    fdf = f.get_group_value_fairness(bdf, tau=tau)
    # Tabla con si se cumplen las medidas de equidad para cada atributo
    tabla_equidad = f.get_group_attribute_fairness(fdf)
    return tabla_equidad
    
if __name__ == '__main__':  

    #data, true_labels = ethik.datasets.load_law_school()  
    data = pd.read_csv('./datos/law_data.csv', index_col=0)    
    
    # Guardamos en un vector todos los atributos protegidos
    protected_attr = ['Asian','Black','Hispanic','Other','White','Male','Female']

    # Obtiene en un diccionario el conjunto de datos y en una partición 80 (train) 20 (test)
    dic_train, dic_test, train, test = preprocess_data(data, protected_attr)
    
    
    # Descomentar si se quieren volver a compilar los modelos guardados
    # crear_borrar_directorio("./datos/modelos", False)
    
    # Obtiene las predicciones para cada modelo
    preds_full_test, preds_full_train, score_full = mod_full(dic_train, dic_test)
    preds_unaware_test, preds_unaware_train, score_unaware = mod_unaware(dic_train, dic_test)
    preds_fair_k_test, preds_fair_k_train, score_fair_k = mod_fair_k(dic_train, dic_test)
    preds_fair_add_test, preds_fair_add_train, score_fair_add = mod_fair_add(dic_train, dic_test)

    # Imprime los valores de RMSE y score resultantes
    print('Full RMSE: %.4f' % np.sqrt(mean_squared_error(preds_full_test, dic_test['FYA'])))
    print('Unaware RMSE: %.4f' % np.sqrt(mean_squared_error(preds_unaware_test, dic_test['FYA'])))
    print('Fair K RMSE: %.4f' % np.sqrt(mean_squared_error(preds_fair_k_test, dic_test['FYA'])))
    print('Fair Add RMSE: %.4f' % np.sqrt(mean_squared_error(preds_fair_add_test, dic_test['FYA'])))
    print('')
    print('Full score: %.4f' % score_full)
    print('Unaware score: %.4f' % score_unaware)
    print('Fair K score: %.4f' % score_fair_k)
    print('Fair Add score: %.4f' % score_fair_add)
    
    # ESTUDIO CON AEQUITAS
    # Leemos el conjunto de datos original creado para el apéndice A
    dataset_orig = preprocess_data_original()
    # Cálculo de las distribuciones del score
    dataset_full = get_aequitas_data(train, preds_full_train, test, preds_full_test)
    dataset_unaware = get_aequitas_data(train, preds_unaware_train, test, preds_unaware_test)
    dataset_fair_k = get_aequitas_data(train, preds_fair_k_train, test, preds_fair_k_test)
    dataset_fair_add = get_aequitas_data(train, preds_fair_add_train, test, preds_fair_add_test)
    # Definimos las paletas de colores
    aq_palette_score = sns.diverging_palette(255, 125, n=2)
    aq_palette_label = sns.diverging_palette(5, 140, n=2)
    # Guardamos las gráficas de distribución del score para los diferentes conjuntos
    
    save_graficas(dataset_full, "label", aq_palette_label, "label_value")
    save_graficas(dataset_full, "score_full", aq_palette_score, "score")
    save_graficas(dataset_unaware, "score_unaware", aq_palette_score, "score")
    save_graficas(dataset_fair_k, "score_fair_k", aq_palette_score, "score")
    save_graficas(dataset_fair_add, "score_fair_add", aq_palette_score, "score")
    save_graficas(dataset_orig, "score_original", aq_palette_score, "score")
    
    # Cálculo de las métricas de grupo
    grupo_full = tabla_metrica_grupo(dataset_full)
    grupo_unaware = tabla_metrica_grupo(dataset_unaware)
    grupo_fair_k = tabla_metrica_grupo(dataset_fair_k)
    grupo_fair_add = tabla_metrica_grupo(dataset_fair_add)
    grupo_orig = tabla_metrica_grupo(dataset_orig)
    # Cálculo de las métricas de sesgo
    attr_ref = {'race':'White', 'sex':'Male'}
    sesgo_full = tabla_metrica_sesgo(dataset_full, attr_ref)
    sesgo_unaware = tabla_metrica_sesgo(dataset_unaware, attr_ref)
    sesgo_fair_k = tabla_metrica_sesgo(dataset_fair_k, attr_ref)
    sesgo_fair_add = tabla_metrica_sesgo(dataset_fair_add, attr_ref)
    sesgo_orig = tabla_metrica_sesgo(dataset_orig, attr_ref)
    # Cálculo de las medidas de equidad
    equidad_full = tabla_medidas_equidad(dataset_full, attr_ref)
    equidad_unaware = tabla_medidas_equidad(dataset_unaware, attr_ref)
    equidad_fair_k = tabla_medidas_equidad(dataset_fair_k, attr_ref)
    equidad_fair_add = tabla_medidas_equidad(dataset_fair_add, attr_ref)
    equidad_orig = tabla_medidas_equidad(dataset_orig, attr_ref)
    #pd.options.display.max_columns = 30
    print(grupo_full)
    print(equidad_full)
    
