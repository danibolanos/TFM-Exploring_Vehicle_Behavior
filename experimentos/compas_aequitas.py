"""
Experimentacion del conjunto de datos COMPAS con Aequitas
Autor: Daniel Bolaños Martínez
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


if __name__ == '__main__':  
    
    # Cargamos el conjunto de datos
    dataset = pd.read_csv('./datos/compas_for_aequitas.csv', index_col=0)  
    
    # Hacemos un estudio previo de los individuos por score y etiqueta real
    aq_palette_score = sns.diverging_palette(255, 125, n=2)
    aq_palette_label = sns.diverging_palette(5, 140, n=2)
    score_race = sns.countplot(x="race", hue="score", 
                data=dataset[dataset.race.isin(['African-American', 'Caucasian', 'Hispanic', 'Other'])], 
                palette=aq_palette_score)
    plt.savefig('./figures/COMPAS/score_race.png')
    plt.clf()
    score_sex = sns.countplot(x="sex", hue="score", data=dataset, palette=aq_palette_score)
    plt.savefig('./figures/COMPAS/score_sex.png')
    plt.clf()
    score_age = sns.countplot(x="age_cat", hue="score", data=dataset, palette=aq_palette_score)
    plt.savefig('./figures/COMPAS/score_age.png')
    plt.clf()
    label_race = sns.countplot(x="race", hue="label_value", 
                data=dataset[dataset.race.isin(['African-American', 'Caucasian', 'Hispanic', 'Other'])], 
                palette=aq_palette_label)
    plt.savefig('./figures/COMPAS/label_race.png')
    plt.clf()
    label_sex = sns.countplot(x="sex", hue="label_value", data=dataset, palette=aq_palette_label)
    plt.savefig('./figures/COMPAS/label_sex.png')
    plt.clf()
    label_sex = sns.countplot(x="age_cat", hue="label_value", data=dataset, palette=aq_palette_label)
    plt.savefig('./figures/COMPAS/label_age.png')
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
    bdf = b.get_disparity_predefined_groups(xtab, original_df=dataset, ref_groups_dict={'race':'Caucasian', 'sex':'Male', 'age_cat':'25 - 45'}, alpha=0.05, mask_significance=True)
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
    fg.savefig('./figures/COMPAS/disparity_group.png')
    m = aqp.plot_fairness_disparity_all(fdf, metrics=['fdr','fpr'], attributes=['race','sex','age_cat'])
    m.savefig('./figures/COMPAS/disparity.png')

    #m = aqp.plot_fairness_disparity(fdf, group_metric='fdr', attribute_name='race', min_group_size=0.01, significance_alpha=0.05)