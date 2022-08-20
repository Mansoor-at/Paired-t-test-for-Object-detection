import numpy as np
from numpy import load
import pandas as pd
from statannot import add_stat_annotation
import seaborn as sns
import matplotlib.pyplot as plt

data_n= load("arrays/10/n_ap.npy")
data_b = load("arrays/10/b_ap.npy")
data_f = load("arrays/10/f_ap.npy")
data_c = load("arrays/10/c_ap.npy")
data_t = load("arrays/10/c_ap.npy")

n= np.expand_dims(data_n,axis=-1)
b= np.expand_dims(data_b,axis=-1)
f= np.expand_dims(data_f,axis=-1)
c = np.expand_dims(data_c,axis=-1)
t = np.expand_dims(data_t,axis=-1)


data = np.concatenate((n,b,f,c,t),axis=-1)
method = ['Proposed',"Baseline", 'UbTeacher_focal',"UbTeacher_ce", "SoftTeacher"]

df_total = pd.DataFrame(columns = ['AP_50','method'])
def add_data(df,data):
    for i in range(5):
        method_name = method[i]
        for j in range(data.shape[0]):
            data_to_append = {}
            #for i in range(len(df.columns)):
            data_to_append[df.columns[0]] = data[j][i]
            data_to_append[df.columns[1]] = method_name
            df = df.append(data_to_append, ignore_index = True)
    return df
df_total = add_data(df_total,data)

sns.set(rc={'figure.figsize':(20,15)})
order = ['Proposed', "Baseline", 'UbTeacher_focal',"UbTeacher_ce","SoftTeacher"]
ax = sns.boxplot(data=df_total, y='AP_50', x='method')
ax, test_results = add_stat_annotation(ax, data=df_total, y='AP_50', x='method',
                                   box_pairs=[("Proposed", "Baseline"),('Proposed', 'UbTeacher_focal'), ("Proposed", 'UbTeacher_ce')
                                              , ("Proposed", "SoftTeacher")],
                                   order=order,test='t-test_paired', text_format='simple', loc='outside',fontsize='xx-large', verbose=2)
plt.xlabel('Method', fontsize=20, labelpad = 15)
plt.ylabel('AP_50', fontsize=20, labelpad = 15)
ax.tick_params(axis='x', which='major', pad=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('result_10.svg', dpi=300, bbox_inches='tight')


