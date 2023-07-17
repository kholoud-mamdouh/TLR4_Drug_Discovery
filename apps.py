import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd
from chembl_webresource_client.new_client import new_client
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)


target = new_client.target
target_query = target.search('TLR4')
targets = pd.DataFrame.from_dict(target_query)
selected_target = targets.target_chembl_id[1]
selected_target

activity = new_client.activity
reselected = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(reselected)
df.to_csv('TLR4_1_bioactivity_data_raw.csv', index=False)
df1 = df[df.standard_value.notna()]
df1 = df1[df.canonical_smiles.notna()]
df1_dropedS = df1.drop_duplicates(['canonical_smiles'])
selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df2 = df1_dropedS[selection]
df2.to_csv('TLR4_2_bioactivity_data_preprocessed.csv', index=False)


df3 = pd.read_csv('TLR4_2_bioactivity_data_preprocessed.csv')

bioactivity_threshold = []
for i in df3.standard_value:
  if float(i) >= 10000:
    bioactivity_threshold.append("inactive")
  elif float(i) <= 1000:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")

bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df4 = pd.concat([df3, bioactivity_class], axis=1)


df4.to_csv('TLR4_3_bioactivity_data_curated.csv', index=False)


sys.path.append('/usr/local/lib/python3.7/site-packages/')

df5=pd.read_csv('TLR4_3_bioactivity_data_curated.csv')

df_no_smiles = df.drop(columns='canonical_smiles')

smiles = []

for i in df.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles,smiles], axis=1)

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

df_lipinski = lipinski(df_clean_smiles.canonical_smiles)


df_combined = pd.concat([df5,df_lipinski], axis=1)

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)

    return x

def norm_value(input):
    norm = []

    for i in input['standard_value']:
       if i > 100000000:
         i = 100000000

       norm.append(i)
    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)

    return x

df_norm = norm_value(df_combined)

df_final = pIC50(df_norm)

df_final.to_csv('TLR4_4_bioactivity_data_3class_pIC50.csv')

df_2class = df_final[df_final['class'] != 'intermediate']
df_2class.to_csv('TLR4_5_bioactivity_data_2class_pIC50.csv')


def mannwhitney(descriptor, verbose=False):
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'class']
  df = df_2class[selection]
  active = df[df['class'] == 'active']
  active = active[descriptor]

  selection = [descriptor, 'class']
  df = df_2class[selection]
  inactive = df[df['class'] == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'

  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results

mannwhitney('pIC50')


import urllib.request
import zipfile

# Download the ZIP file
url = 'https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip'
urllib.request.urlretrieve(url, 'padel.zip')

# Unzip the file
with zipfile.ZipFile('padel.zip', 'r') as zip_ref:
    zip_ref.extractall()


df6 = pd.read_csv("TLR4_4_bioactivity_data_3class_pIC50.csv")

selection = ['canonical_smiles','molecule_chembl_id']
df6_selection = df6[selection]
df6_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

df6_X = pd.read_csv('descriptors_output.csv')
df6_X = df6_X.drop(columns=['Name'])

df6_Y = df6['pIC50']

dataset = pd.concat([df6_X,df6_Y], axis=1)

dataset.to_csv('TLR4_6_bioactivity_data_3class_pIC50_pubchem_fp.csv', index=False)

df7 = pd.read_csv('TLR4_6_bioactivity_data_3class_pIC50_pubchem_fp.csv')

df8=df7.dropna(axis=0)

X = df8.drop('pIC50', axis=1)
Y = df8.pIC50

from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)

    return input_data[input_data.columns[selection.get_support(indices=True)]]
X = pd.DataFrame(X)
X = remove_low_variance(X, threshold=0.1)

X.to_csv('descriptor_list.csv', index = False)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X, Y)




st.title("Predictor")
st.image("1.jpeg",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(X)
    
    graph = st.selectbox("What kind of Graph ? ",["active"])
    if graph == "active":
      plt.figure(figsize=(5.5, 5.5))
      sns.countplot(x='class', data=df_2class, edgecolor='black')
      sns.countplot(x='class', data=df_2class, edgecolor='black')
      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('Frequency', fontsize=14, fontweight='bold')

      plt.savefig('plot_bioactivity_class.pdf')
      plt.tight_layout()
      st.pyplot()
      """#Scatter plot of MW versus LogP"""

      plt.figure(figsize=(5.5, 5.5))

      sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', edgecolor='black', alpha=0.7)
      plt.xlabel('MW', fontsize=14, fontweight='bold')
      plt.ylabel('LogP', fontsize=14, fontweight='bold')
      plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
      plt.savefig('plot_MW_vs_LogP.pdf')
      plt.tight_layout()
      st.pyplot()

      plt.figure(figsize=(5.5, 5.5))
      sns.boxplot(x = 'class', y = 'pIC50', data = df_2class)

      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

      plt.savefig('plot_ic50.pdf')
      plt.tight_layout()
      st.pyplot()


      """#MW"""

      plt.figure(figsize=(5.5, 5.5))

      sns.boxplot(x = 'class', y = 'MW', data = df_2class)
      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('MW', fontsize=14, fontweight='bold')

      plt.savefig('plot_MW.pdf')
      plt.tight_layout()
      st.pyplot()
      mannwhitney('MW')

      """#LogP"""

      plt.figure(figsize=(5.5, 5.5))

      sns.boxplot(x = 'class', y = 'LogP', data = df_2class)
      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('LogP', fontsize=14, fontweight='bold')

      plt.savefig('plot_LogP.pdf')
      plt.tight_layout()
      st.pyplot()
      mannwhitney('LogP')

      """#NumHDonors"""

      plt.figure(figsize=(5.5, 5.5))
      sns.boxplot(x = 'class', y = 'NumHDonors', data = df_2class)
      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

      plt.savefig('plot_NumHDonors.pdf')
      plt.tight_layout()
      st.pyplot()
      """Statistical analysis | Mann-Whitney U Test"""

      mannwhitney('NumHDonors')

      """#NumHAcceptors"""

      plt.figure(figsize=(5.5, 5.5))
      sns.boxplot(x = 'class', y = 'NumHAcceptors', data = df_2class)
      plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
      plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

      plt.savefig('plot_NumHAcceptors.pdf')
      plt.tight_layout()
      st.pyplot()
      mannwhitney('NumHAcceptors')

    
if nav == "Prediction":
    st.header("the percentage of active  ")
    #val1 = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    #val2 = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    #val3 = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    #val4 = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    #val5 = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
  
    pred =model.predict(X)
    print(pred)
    pred = np.round(pred, decimals=2)
    if st.button("Predict"):
        st.success(f"the percentage of active is {pred}")


