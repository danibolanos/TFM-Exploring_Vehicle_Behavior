"""
Generación de gráficas para el contrafactual para el modelo del escenario 2
replicado para Counterfactual Fairness (Kusner et al. 2017).

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
import matplotlib.pyplot as plt
import copy
import pymc3 as pm


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
    dic_test = create_pystan_dic(test, protected_attr, False)

    return dic_train, dic_test, train_orig, test_orig

# Modelo Total: usa todos los atributos para la predicción
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
    preds = lr_full.predict(x_test)

    return preds


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
    preds = lr_unaware.predict(x_test)

    return preds

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

# Implementación del método MCMC con la biblioteca pymc3 usando theano
def MCMC_THEANO(data, path_model, train=True, samples=2000):
    model_fit = Path(path_model)
    
    # Comprobamos si ya existe un archivo con el modelo entrenado
    if model_fit.is_file():
        file = open(path_model, "rb")
        samples = pickle.load(file)
    else:
        C = data['C']
        N = data['N']
        A = data['A']
        # Definimos el constructor para el modelo
        model = pm.Model()
    
        with model:
            # Parámetros dependientes de las variables observadas
            K = pm.Normal("K", mu=0, sigma=1, shape=(1, N))
            gpa0 = pm.Normal("gpa0", mu=0, sigma=1)
            lsat0 = pm.Normal("lsat0", mu=0, sigma=1)
            w_k_gpa = pm.Normal("w_k_gpa", mu=0, sigma=1)
            w_k_lsat = pm.Normal("w_k_lsat", mu=0, sigma=1)
            w_k_fya = pm.Normal("w_k_fya", mu=0, sigma=1)
            w_a_gpa = pm.Normal("w_a_gpa", mu=np.zeros(C), sigma=np.ones(C), shape=C)
            w_a_lsat = pm.Normal("w_a_lsat", mu=np.zeros(C), sigma=np.ones(C), shape=C)
            w_a_fya = pm.Normal("w_a_fya", mu=np.zeros(C), sigma=np.ones(C), shape=C)
    
            sigma_gpa_2 = pm.InverseGamma("sigma_gpa_2", alpha=1, beta=1)
    
            mu = gpa0 + (w_k_gpa * K) + pm.math.dot(A, w_a_gpa)
    
            # Definidos las variables observadas
            gpa = pm.Normal("gpa", mu=mu, sigma=pm.math.sqrt(sigma_gpa_2),
                observed=list(data["GPA"]), shape=(1, N))
            lsat = pm.Poisson("lsat", pm.math.exp(lsat0 + w_k_lsat * K + pm.math.dot(A, w_a_lsat)),
                observed=list(data["LSAT"]), shape=(1, N))
            if train:
                fya = pm.Normal("fya", mu=w_k_fya * K + pm.math.dot(A, w_a_fya), sigma=1,
                     observed=list(data["FYA"]), shape=(1, N))
            step = pm.Metropolis()
            samples = pm.sample(samples, step)
            # Guardamos el modelo entrenado
            file = open(path_model, "wb")
            pickle.dump(samples, file, protocol=-1)

    return samples


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

# Modelo no determinista: suponemos variable de ruido 'K' para generar la distribución del resto
def fair_learning(dic_train, dic_test, protected_attr=[], refobj=[], change=False, num=""):
    modelos_dir = Path("./datos/modelos_prueba")
    
    # Comprobamos si ya existe un archivo con el modelo entrenado
    if not modelos_dir.exists():
        crear_borrar_directorio(modelos_dir, True)
        
    # Entrenamos el modelo utilizando el MCMC y obtenemos las muestras para cada punto
    #train_samples = MCMC(dic_train, "./datos/modelos_prueba/train_model"+num+".pkl", "./datos/stan/law_school_train.stan")
    # Ejecución con theano usando la algoritmo Metropoli para MCMC
    train_samples = MCMC_THEANO(dic_train, "./datos/modelos_prueba/train_model"+num+".pkl")
    # Obtenemos la media de la variable K para train
    train_k = np.mean(train_samples["K"], axis=0).reshape(-1, 1)
    
    '''
    # Muestra la traza para las muestras de K
    k_trace = train_samples["K"].reshape(-1, 1)
    plt.subplot(1, 2, 1)
    plt.hist(k_trace, bins=100)
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(k_trace)), k_trace, s=1)
    plt.savefig('./figures/muestrak1.png')
    plt.show()
    '''
    
    # Usamos la distribución de K aprendida y hacemos las medias del resto de variables para test
    #dic_means = get_mean_params(train_samples, dic_test)
    dic_means=dic_test
    
    # Si change=True hacemos el cambio de los individuos previo
    if change:
        dic_means = cambiar_individuos(dic_means, protected_attr, refobj[0], refobj[1])
        
    # Volvemos a inferir sobre el modelo esta vez usando el modelo sin FYA para test
    test_samples = MCMC_THEANO(dic_means, "./datos/modelos_prueba/test_model"+num+".pkl", False)
    #test_samples = MCMC(dic_means, "./datos/modelos_prueba/test_model"+num+".pkl", "./datos/stan/law_school_only_k.stan")
    # Obtenemos la media de la variable K para test
    test_k = np.mean(test_samples["K"], axis=0).reshape(-1, 1)
    
    return train_k, test_k


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
    preds = lr_fair_k.predict(x_test)
    
    return preds


# Devuelve un vector de predicciones de unos individuos en concreto
def preds_individuos(original_data, protected_attr, atributo, preds):
    pos_attr = protected_attr.index(atributo)
    individuos = original_data['A']
    preds_ind = []
    for i in range(0,len(individuos)):
        if individuos[i][pos_attr] == 1:
            preds_ind.append(preds[i])

    return preds_ind

# Mismo comportamiento que la función anterior pero para un array atributos
def preds_array_individuos(original_data, protected_attr, atributos, preds):
    preds_ind = []
    for i in range(0, len(atributos)):
        preds_attr_i = preds_individuos(original_data, protected_attr, atributos[i], preds)
        preds_ind = np.hstack((preds_ind, preds_attr_i))
        
    return preds_ind

# Cambia el atributo objetivo por el valor del referencia para los individuos que lo cumplen
def cambiar_individuos(dic, protected_attr, referencia, objetivo):
    dic_data = copy.deepcopy(dic)
    # Guardamos el índice referente al atributo sensible 
    referencia = protected_attr.index(referencia)
    objetivo = protected_attr.index(objetivo)
    # Si es el individuo es del tipo atributo objetivo se cambia al referencia
    for i in range(0,len(dic_data['A'])):
        if dic_data['A'][i][objetivo] == 1:
            dic_data['A'][i][referencia] = 1
            dic_data['A'][i][objetivo] = 0
    
    return dic_data

# Modelo Full para la construcción de la gráfica
def mod_full_k(dic_tr, dic_te, protected_attr, refobj=[], num="", change=False):    
    # A partir del modelo determinista genero las distribuciones de K
    train_k, test_k = fair_learning(dic_train, dic_test, protected_attr, refobj, change, num)

    # Construimos los nuevos conjuntos de datos sobre los que predecir
    x_train = train_k
    x_test = test_k
    
    y = dic_train['FYA']

    # Entrenamiento del modelo sobre el conjunto x_train
    lr_full = LinearRegression()
    lr_full.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_full.predict(x_test)
    
    preds_array = preds_array_individuos(dic_te, protected_attr, refobj, preds)

    return preds_array

# Modelo Unaware para la construcción de la gráfica
def mod_unaware_k(dic_tr, dic_te, protected_attr, refobj=[], num="", change=False):
    dic_train = copy.deepcopy(dic_tr)
    dic_test = copy.deepcopy(dic_te)
    
    # A partir del modelo determinista genero las distribuciones de K
    train_k, test_k = fair_learning(dic_train, dic_test, num, change)

    # Construimos los nuevos conjuntos de datos sobre los que predecir
    x_train = train_k
    x_test = test_k
    
    y = dic_train['FYA']

    # Entrenamiento del modelo sobre el conjunto x_train
    lr_full = LinearRegression()
    lr_full.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_full.predict(x_test)
    
    preds_array = preds_array_individuos(dic_te, protected_attr, refobj, preds)

    return preds_array

def save_graphics(datas, name):
    graph = pd.concat({'black<->white': datas[0], 'asian<->white': datas[1], 'hispanic<->white': datas[2], 'female<->male': datas[3]})
    graph = graph.reset_index(level=0).rename(columns={'level_0': 'subplot'}).reset_index()
    g = sns.displot(kind='kde', data=graph, x='FYA', hue='', fill=True, col='subplot', col_wrap=4)
    for ax, col_name in zip(g.axes.flat, g.col_names):
        ax.set_title(col_name)
    plt.savefig(name)
    plt.show()
    
if __name__ == '__main__':  

    #data, true_labels = ethik.datasets.load_law_school()  
    data = pd.read_csv('./datos/law_data.csv', index_col=0)    
    
    # Guardamos en un vector todos los atributos protegidos
    protected_attr = ['Asian','Black','Hispanic','Other','White','Male','Female']

    # Obtiene en un diccionario el conjunto de datos y en una partición 80 (train) 20 (test)
    dic_train, dic_test, train_o, test_o = preprocess_data(data, protected_attr)
    
    # Descomentar si se quieren volver a compilar los modelos guardados
    # crear_borrar_directorio("./datos/modelos_prueba", False)

    # Obtiene las predicciones para cada modelo
    preds_full = mod_full(dic_train, dic_test)
    preds_unaware = mod_unaware(dic_train, dic_test)
    preds_fair_k = mod_fair_k(dic_train, dic_test)

    # Imprime los valores de RMSE resultantes
    print('Full RMSE: %.4f' % np.sqrt(mean_squared_error(preds_full, test_o['ZFYA'])))
    print('Unaware RMSE: %.4f' % np.sqrt(mean_squared_error(preds_unaware, test_o['ZFYA'])))
    print('Fair K RMSE: %.4f' % np.sqrt(mean_squared_error(preds_fair_k, test_o['ZFYA'])))
    
    # GRÁFICA
    # FILA MODELO FULL
    preds_prueba1 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Black'], "original_1")
    preds_prueba_c1 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Black'], "counter_1", True)
    preds_prueba2 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Asian'], "original_2")
    preds_prueba_c2 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Asian'], "counter_2", True)
    preds_prueba3 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Hispanic'], "original_3")
    preds_prueba_c3 = mod_full_k(dic_train, dic_test, protected_attr, ['White', 'Hispanic'], "counter_3", True)
    preds_prueba4 = mod_full_k(dic_train, dic_test, protected_attr, ['Male', 'Female'], "original_4")
    preds_prueba_c4 = mod_full_k(dic_train, dic_test, protected_attr, ['Male', 'Female'], "counter_4", True)
    
    # FILA MODELO UNAWARE
    preds_prueba5 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Black'], "original_5")
    preds_prueba_c5 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Black'], "counter_5", True)
    preds_prueba6 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Asian'], "original_6")
    preds_prueba_c6 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Asian'], "counter_6", True)
    preds_prueba7 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Hispanic'], "original_7")
    preds_prueba_c7 = mod_unaware_k(dic_train, dic_test, protected_attr, ['White', 'Hispanic'], "counter_7", True)
    preds_prueba8 = mod_unaware_k(dic_train, dic_test, protected_attr, ['Male', 'Female'], "original_8")
    preds_prueba_c8 = mod_unaware_k(dic_train, dic_test, protected_attr, ['Male', 'Female'], "counter_8", True)
    
    # Dibujamos la gráfica de densidad de las predicciones de FYA
    sns.set(style="darkgrid")
    df = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba1, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c1, '': 'counterfactual'})])
    df1 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba2, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c2, '': 'counterfactual'})])
    df2 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba3, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c3, '': 'counterfactual'})])
    df3 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba4, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c4, '': 'counterfactual'})])

    df4 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba5, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c5, '': 'counterfactual'})])
    df5 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba6, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c6, '': 'counterfactual'})])
    df6 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba7, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c7, '': 'counterfactual'})])
    df7 = pd.concat(axis=0, ignore_index=True, objs=[
        pd.DataFrame.from_dict({'FYA': preds_prueba8, '': 'original data'}),
        pd.DataFrame.from_dict({'FYA': preds_prueba_c8, '': 'counterfactual'})])    
    
    save_graphics([df, df1, df2, df3], './figures/grafos/grafo_full2.png')
    save_graphics([df4, df5, df6, df7], './figures/grafos/grafo_unaware2.png')
