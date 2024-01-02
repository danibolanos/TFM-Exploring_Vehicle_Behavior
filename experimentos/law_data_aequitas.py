"""
Experimentacion del conjunto de datos law_data con Aequitas
Autor: Daniel Bolaños Martínez
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


# Preprocesamiento del conjunto de datos 'law_data.csv'
def preprocess_data():
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
    
    # Reordenamos las columnas
    cols = dataset.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    dataset = dataset[cols] 
    
    dataset.to_csv("./datos/law_for_aequitas.csv", index=False)

    return dataset

if __name__ == '__main__':  

    # Obtiene en un diccionario el conjunto de datos y en una partición 80 (train) 20 (test)
    dataset = preprocess_data()

    # Hacemos un estudio previo de los individuos por score y etiqueta real
    aq_palette_score = sns.diverging_palette(255, 125, n=2)
    aq_palette_label = sns.diverging_palette(5, 140, n=2)
    score_race = sns.countplot(x="race", hue="score", 
                data=dataset[dataset.race.isin(['Black', 'White', 'Hispanic'])], palette=aq_palette_score)
    plt.savefig('./figures/LAW_DATA/score_race_law.png')
    plt.clf()
    score_sex = sns.countplot(x="sex", hue="score", data=dataset, palette=aq_palette_score)
    plt.savefig('./figures/LAW_DATA/score_sex_law.png')
    plt.clf()
    label_race = sns.countplot(x="race", hue="label_value", 
                data=dataset[dataset.race.isin(['Black', 'White', 'Hispanic'])], palette=aq_palette_label)
    plt.savefig('./figures/LAW_DATA/label_race_law.png')
    plt.clf()
    label_sex = sns.countplot(x="sex", hue="label_value", data=dataset, palette=aq_palette_label)
    plt.savefig('./figures/LAW_DATA/label_sex_law.png')
    plt.clf()
    
    # Calculamos la tabla de metricas de grupo
    g = Group()
    xtab, _ = g.get_crosstabs(dataset)
    absolute_metrics = g.list_absolute_metrics(xtab)
    # Mostramos la tabla por pantalla
    tabla_grupo = xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
    print(tabla_grupo)
 
    aqp = Plot()    
    # Plot de los valores de las metricas de grupo para FNR
    fnr = aqp.plot_group_metric(xtab, 'fnr')
    # Plot de los valores de las metricas de grupo para FNR eliminando poblaciones con umbral de individuos
    fnr = aqp.plot_group_metric(xtab, 'fnr', min_group_size=0.05)
    # Metricas de grupo para todas las elegidas
    p = aqp.plot_group_metric_all(xtab, metrics=['ppr','pprev','fnr','fpr'], ncols=4)
    
    # Calculamos las metricas de sesgo
    b = Bias()
    # Establecemos los atributos de referencia
    bdf = b.get_disparity_predefined_groups(xtab, original_df=dataset, ref_groups_dict={'race':'White', 'sex':'Male'}, alpha=0.05, mask_significance=True)
    calculated_disparities = b.list_disparities(bdf)
    disparity_significance = b.list_significance(bdf)
    # Mostramos la tabla de metricas de sesgo
    print(bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance])
    
    # Plots de disparidad
    #aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='race', significance_alpha=0.05)
    #j = aqp.plot_disparity_all(bdf, metrics=['precision_disparity', 'fpr_disparity'], attributes=['age_cat'], significance_alpha=0.05)

    # Definimos las medidas de equidad a partir de la tabla de metricas de sesgo
    f = Fairness()
    # Establecemos el valor del umbral con la variable tau
    fdf = f.get_group_value_fairness(bdf, tau=0.8)
    #parity_detrminations = f.list_parities(fdf)
    # Tabla con si se cumplen las medidas de equidad para cada atributo
    gaf = f.get_group_attribute_fairness(fdf)
    #print(gaf['Equalized Odds'])
    
    # Metricas de grupo y de sesgo una vez aplicados los umbrales de equidad
    fg = aqp.plot_fairness_group_all(fdf, ncols=2, metrics = ['ppr','pprev','fdr','fpr','for','fnr'])
    fg.savefig('./figures/LAW_DATA/disparity_group_law.png')
    m = aqp.plot_fairness_disparity_all(fdf, metrics=['for','fnr'], attributes=['race'])
    m.savefig('./figures/LAW_DATA/disparity_law_race.png')
    m = aqp.plot_fairness_disparity_all(fdf, metrics=['for','fnr'], attributes=['sex'])
    m.savefig('./figures/LAW_DATA/disparity_law_sex.png')

    #m = aqp.plot_fairness_disparity(fdf, group_metric='for', attribute_name='race', min_group_size=0.01, significance_alpha=0.05)
    #m.savefig('./figures/disparity_law.png')