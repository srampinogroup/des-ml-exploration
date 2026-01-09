import sys


import pandas             as pd
import numpy              as np
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
import seaborn            as sns
import itertools
import time
import shap


from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from math                    import sqrt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics         import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from xgboost                 import XGBRegressor
from catboost                import CatBoostRegressor
from sklearn.svm             import SVR
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.metrics         import mean_squared_error
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neural_network  import MLPRegressor
from matplotlib.colors       import ColorConverter
from prettytable             import PrettyTable

from scipy.stats             import shapiro
from scipy.stats             import jarque_bera


def column_name_OG(df):
  df_renamed = df.rename(columns={
      'X#1 (molar fraction)': 'Molar fraction of component#1',
      '1': 'Type I',
      '2': 'Type II',
      '3': 'Type III',
      '4': 'Type IV',
      '5': 'Type V',
      'IL': 'Type IL',
      'Tmelt#1': 'Melting temperature of component#1, K',
      'Tmelt#2': 'Melting temperature of component#2, K',
      'MW': 'Molecular weight, g/mol',
      'HBD_total': 'Number of hydrogen bond donors',
      'fr_Al_COO': 'Number of aliphatic carboxylic acids',
      'fr_Ar_COO': 'Number of aromatic carboxylic acide',
      'fr_Ar_N': 'Number of aromatic nitrogens',
      'fr_Ar_OH': 'Number of aromatic hydroxyl groups',
      'fr_NH0': 'Number of Tertiary amines',
      'fr_NH1': 'Number of Secondary amines',
      'fr_amide': 'Number of amides',
      'AROM': 'Number of aromatic rings',
      'ALERTS': 'Chemical toxicity evaluation',
      'n_HM': 'Number of heavy atoms'
  })
  return df_renamed

def column_name_NEW(df):
  df_renamed = df.rename(columns={
      'Tmelt, K': 'Melting temperature, K',
      'X#1 (molar fraction)': 'Molar fraction of component#1',
      'T#1': 'Temperature condition #1',
      'T#2': 'Temperature condition #2',
      'MW': 'Molecular weight, g/mol',
      'HBD': 'Number of hydrogen bond donors',
      'fr_Al_COO': 'Number of aliphatic carboxylic acids',
      'fr_Ar_COO': 'Number of aromatic carboxylic acids',
      'fr_Ar_N': 'Number of aromatic nitrogens',
      'fr_Ar_OH': 'Number of aromatic hydroxyl groups',
      'fr_NH0': 'Number of Tertiary amines',
      'fr_NH1': 'Number of Secondary amines',
      'fr_amide': 'Number of amides',
      'AROM': 'Number of aromatic rings',
      'ALERTS': 'Chemical toxicity evaluation',
      'n_HM': 'Number of heavy atoms'
  })
  return df_renamed
  
  
  

def feature_importance_tree(model_name, model, x_test_scaled, df_renamed):
    list_columns = df_renamed.loc[:, 'Molar fraction of component#1' : 'Number of heavy atoms'].columns
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test_scaled)
    shap.summary_plot(shap_values,x_test_scaled,plot_type='bar',color=colors[0],feature_names=list_columns,max_display=10,show=False )
    plt.title(f'Feature importance {model_name}', fontsize=14, pad=20)

    plt.savefig(f'out/Feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 




def feature_importance_kernel(model_name, model, x_test_scaled, x_train_scaled, df_renamed):
    list_columns = df_renamed.loc[:, 'Molar fraction of component#1' : 'Number of heavy atoms'].columns
    background_data = pd.DataFrame(x_train_scaled).sample(100, random_state=42)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(x_test_scaled)
    shap.summary_plot(shap_values, x_test_scaled, plot_type='bar', color=colors[0], feature_names=list_columns, max_display = 10, show=False)
    plt.title(f'Feature importance {model_name}', fontsize=14, pad=20)

    plt.savefig(f'out/Feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return



def read_csv(file_name):
  df = pd.read_csv(file_name)
  #print(df)
  #print(df.columns)
  return df


def df_foranalysis(melt_temp_df):
  melt_temp_df = melt_temp_df.round({'X#1 (molar fraction)': 3, 'X#2 (molar fraction)': 3}) # Rounding of molar fractions
  melt_temp_df.rename(columns = {'Tmelt, K':'Melting temperature',}, inplace = True )
  return (melt_temp_df)

def DESclassification(melt_temp_df):
  bin_ter = melt_temp_df[['Number of components', 'Type of DES', 'Melting temperature']].groupby(['Number of components', 'Type of DES']).agg('count')
  bin_ter

  colors = ['#a5678e', '#beb7d9', '#31539d','#e8b7d4', '#7eabd4', 'gray'] # Palette of colors
  fig, ax = plt.subplots(figsize = (12, 8)) #Initialization - creating an empty graph
  labels = ['Binary DES,\n Type I','Binary DES, \n Type II', 'Binary DES, \n Type III', 'Binary DES, \n Type IV', 'Binary DES,\n Type V', 'Ionic liquids']
  ax.pie(bin_ter['Melting temperature'], colors = colors, autopct = '%1.2f%%', textprops={'fontsize': 30}, explode=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], pctdistance=1.23, radius=1) #Change names, colors, add%, increase font
  plt.legend(
      bbox_to_anchor = (-0.3, 0.45, 0.25, 0.25),
      loc = 'best', labels = labels, fontsize = 30)
  ax.set_title('The ratio of different types od DES', fontsize = 30, y=1.05) #Title of a separate chart
  plt.savefig("datanalysis/ratio_of_DES_type.png", dpi=300)
  plt.close(fig)
  return



def popular_components(melt_temp_df):

  #Table for the number of unique systems in various articles
  max_comp = melt_temp_df[melt_temp_df['Number of components'] == 2][['Component#1','Component#2', 'Reference (DOI)']].groupby(['Component#1','Component#2']).agg('nunique')
  max_comp = max_comp.sort_values('Reference (DOI)', ascending = False)
  #To understand the number of times a particular acceptor participates in various systems, we reset the indexes and regroup the samples
  max_comp.reset_index(level=[0,1], inplace = True)
  #Regrouping
  max_acceptor = max_comp[['Component#1', 'Reference (DOI)']].groupby('Component#1').agg('sum')
  max_acceptor = max_acceptor.sort_values('Reference (DOI)', ascending = False)
  max_acceptor['Component#1'] = max_acceptor.index
  f_get_upper = lambda x: x.upper()
  max_acceptor['Component#1'] = max_acceptor['Component#1'].apply(f_get_upper)
  max_acceptor.index = max_acceptor['Component#1']
  top_acceptors = max_acceptor.head(10)
  top_acceptors.to_csv('datanalysis/top_acceptors.csv', index=False)
  

  fig, ax = plt.subplots(figsize = (12, 8)) # Chart initialization
  ax.barh(max_acceptor.head(8).index, max_acceptor.head(8)['Reference (DOI)'], color = colors) # Building a bar chart
  ax.set_title('The most popular acceptors of binary systems', fontsize = 30) # Title
  ax.set_xlabel('Quantity', fontsize = 20) # X-axis signature
  ax.set_yticks(max_acceptor.index[0:8])
  ax.tick_params(axis='both', which='major', labelsize=15)
  plt.savefig("datanalysis/popular_acceptors.png", dpi=300)
  plt.close(fig)
  
    
  #To understand the number of times a particular donor participates in various systems, we reset the indexes and regroup the samples
  max_donor = max_comp[['Component#2', 'Reference (DOI)']].groupby('Component#2').agg('sum')
  max_donor = max_donor.sort_values('Reference (DOI)', ascending = False) #Sort in descending order
  max_donor['Component#2'] = max_donor.index
  f_get_upper = lambda x: x.upper()
  max_donor['Component#2'] = max_donor['Component#2'].apply(f_get_upper)
  max_donor.index = max_donor['Component#2']
  top_donors = max_donor.head(10)
  top_donors.to_csv('datanalysis/top_donors.csv', index=False)
  
  
  fig, ax = plt.subplots(figsize = (12, 8)) # Chart initialization
  ax.barh(max_donor.head(8).index, max_donor.head(8)['Reference (DOI)'], color = colors) # Building a bar chart
  ax.set_title('The most popular donors of binary systems', fontsize = 30) # Title
  ax.set_xlabel('Quantity', fontsize = 20) # X-axis signature
  ax.tick_params(axis='both', which='major', labelsize=15)
  plt.savefig("datanalysis/popular_donors.png", dpi=300)
  plt.close(fig)
  
  
  popular_system = melt_temp_df[melt_temp_df['Number of components'] == 2][['Component#1', 'Component#2', 'Melting temperature']].groupby(['Component#1', 'Component#2']).agg('count')
  popular_system = popular_system.sort_values('Melting temperature', ascending = False) #Sort in descending order
  top_systems = popular_system.head(10)
  top_systems.to_csv('datanalysis/popular_systems.csv')
  
  return

def melting_temperature_distribution(melt_temp_df):

  fig_time, ax_time = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax_time[0].hist(melt_temp_df['Melting temperature'], color = colors[2], bins = 13) # Building a histogram
  ax_time[0].set_title('Melting temperature distribution', fontsize=40) #Title
  ax_time[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax_time[0].set_ylabel('Sample quantity', fontsize=40) #Y-axis signature
  ax_time[0].tick_params(axis='both', which='major', labelsize=40)
  ax_time[1]= sns.boxplot(y = melt_temp_df['Melting temperature'], color= colors[2]) #Building a boxplot
  ax_time[1].set_title('Boxplot for  Melting temperature for all values', fontsize=40) #Title
  ax_time[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax_time[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/MT_distribution.png", dpi=300)
  plt.close(fig_time)
 
  data = melt_temp_df['Melting temperature']
  stat, p = shapiro(data)
  print
  print(stat, p)
  print('stat={0:.3f}, p={0:.3f}' .format(stat, p))
  if p > 0.05:
      print(p)
      print('Probably Gaussian')
  else:
      print(p)
      print('Probably not Gaussian')
  print(jarque_bera(data))

  return

def distribution_by_type(melt_temp_df):
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == '1']['Melting temperature'], color = colors[0]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for I type of binary DES', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature, cP', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == '1']['Melting temperature'], color = colors[0]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for I type of binary DES', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/TypeI_distribution.png", dpi=300)
  plt.close(fig)
  
  
  
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == '2']['Melting temperature'], color = colors[1]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for II type of binary DES', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == '2']['Melting temperature'], color = colors[1]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for II type of binary DES', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/TypeII_distribution.png", dpi=300)
  plt.close(fig)
  
  
  
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == '3']['Melting temperature'], color = colors[2]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for III type of binary DES', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == '3']['Melting temperature'], color = colors[2]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for III type of binary DES', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/TypeIII_distribution.png", dpi=300)
  plt.close(fig)
  
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == '4']['Melting temperature'], color = colors[3]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for IV type of binary DES', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == '4']['Melting temperature'], color = colors[3]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for IV type of binary DES', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/TypeIV_distribution.png", dpi=300)
  plt.close(fig)
  
  
  
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == '5']['Melting temperature'], color = colors[4]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for V type of binary DES', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == '5']['Melting temperature'], color = colors[4]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for V type of binary DES', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/TypeV_distribution.png", dpi=300)
  plt.close(fig)
  
  
  
  fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12)) # Initialization 
  ax[0].hist(melt_temp_df[melt_temp_df['Type of DES'] == 'IL']['Melting temperature'], color = colors[5]) # Building a histogram
  ax[0].set_title('Melting temperature distribution\n for Ionic liquids', fontsize=40) #Title
  ax[0].set_xlabel('Melting temperature', fontsize=40) #X-axis signature
  ax[0].set_ylabel('Quantity', fontsize=40) #Y-axis signature
  ax[0].tick_params(axis='both', which='major', labelsize=40)
  ax[1] = sns.boxplot(y = melt_temp_df[melt_temp_df['Type of DES'] == 'IL']['Melting temperature'], color = colors[5]) #Building a boxplot
  ax[1].set_title('Boxplot for Melting temperature\n for Ionic liquids', fontsize=40) #Title
  ax[1].set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax[1].tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/IL_distribution.png", dpi=300)
  plt.close(fig)
  
  
  fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (24, 12)) # Initialization
  ax = sns.violinplot(y = melt_temp_df['Melting temperature'], x = melt_temp_df['Type of DES'], hue = melt_temp_df['Type of DES'], legend = False, palette = colors) #Building a violinplot 
  ax.set_xticks([0, 1,2,3,4, 5], labels= ['Binary DES,\n Type I', 'Binary DES, \n Type II', 'Binary DES, \n Type III', 'Binary DES, \n Type IV', 'Binary DES,\n Type V', 'Ionic liquids'])
  ax.set_title('Violinplot for different types of DES', fontsize=40) #Title
  ax.set_ylabel('Melting temperature', fontsize=40) #Y-axis signature
  ax.set_xlabel('Types of DESs', fontsize=40) #X-axis signature
  ax.tick_params(axis='both', which='major', labelsize=40)
  plt.savefig("datanalysis/AllType_distribution.png", dpi=300)
  plt.close(fig)
   
  return
  

def df_preprocessing(df):
  df_unique = df[['Component#1', 'Component#2']].drop_duplicates()
  df_unique = df_unique.reset_index(drop=True)
  df_unique = df_unique.reset_index()  # convert indexes to a column
  df_unique['index_col'] = df_unique['index']  # create a new column and assign it the values of the column with indexes
  # combine df2 and df_unique by Component#1 and Component#2 columns
  merged = df.merge(df_unique, on=['Component#1', 'Component#2'])
  # create the indexes column and fill it with values from the index column from df_unique
  merged['indexes'] = merged['index']
  # we remove the index column from merged, since we no longer need it
  merged.drop('index', axis=1, inplace=True)
  df = merged.drop(['Number of components', 'Component#1', 'Component#2', 'X#2 (molar fraction)',
                    'Phase diagram (Yes/No)', 'Reference (DOI)', 'Smiles#1', 'Smiles#2', 'index_col'], axis=1)
  # moving the column 'Tmelt, K' to the first place
  df = df[['Tmelt, K', 'X#1 (molar fraction)', 'Type of DES', 'Tmelt#1', 'Tmelt#2', 'MW', 'HBD_total', 'fr_Al_COO', 'fr_Ar_COO',
           'fr_Ar_N', 'fr_Ar_OH', 'fr_NH0', 'fr_NH1', 'fr_amide', 'AROM', 'ALERTS', 'n_HM', 'indexes']]
  df = df.rename(columns={'indexes': 'ind_syst'})
  #print(df.columns)
  return df

def df_fitprocessing(df):
  # creating a DataFrame df_unique with a list of unique combinations from component columns
  df_unique = df[['Component#1','Component#2']].drop_duplicates()
  df_unique = df_unique.reset_index(drop=True)
  df_unique = df_unique.reset_index()  # convert indexes to a column
  df_unique['index_col'] = df_unique['index']  # create a new column and assign it the values of the column with indexes
  # combine df2 and df_unique by Component#1 and Component#2 columns
  merged = df.merge(df_unique, on=['Component#1', 'Component#2'])
  # create the indexes column and fill it with values from the index column from df_unique
  merged['indexes'] = merged['index']
  # we remove the index column from merged, since we no longer need it
  merged.drop('index', axis=1, inplace=True)
  
  #Adapted columns to new dataframe 
  df = merged.drop(['Component#1', 'Component#2', 'X#2 (molar fraction)',
                    'Smiles#1', 'Smiles#2', 'index_col'], axis=1)
  
  #Adapted columns to new dataframe 
  df = df[['Tmelt, K', 'X#1 (molar fraction)','T#1',
         'T#2', 'MW', 'HBD', 'fr_Al_COO', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_OH',
         'fr_NH0', 'fr_NH1', 'fr_amide', 'AROM', 'ALERTS', 'n_HM', 'indexes']]
   
  df = df.rename(columns={'indexes': 'ind_syst'})
  return df

def process_no_tod(df):
  df_unique = df[['Component#1', 'Component#2']].drop_duplicates()
  df_unique = df_unique.reset_index(drop=True)
  df_unique = df_unique.reset_index()  # convert indexes to a column
  df_unique['index_col'] = df_unique['index']  # create a new column and assign it the values of the column with indexes
  # combine df2 and df_unique by Component#1 and Component#2 columns
  merged = df.merge(df_unique, on=['Component#1', 'Component#2'])
  # create the indexes column and fill it with values from the index column from df_unique
  merged['indexes'] = merged['index']
  # we remove the index column from merged, since we no longer need it
  merged.drop('index', axis=1, inplace=True)
  df = merged.drop(['Number of components', 'Type of DES', 'Component#1', 'Component#2', 'X#2 (molar fraction)',
                    'Phase diagram (Yes/No)', 'Reference (DOI)', 'Smiles#1', 'Smiles#2', 'index_col'], axis=1)
  # moving the column 'Tmelt, K' to the first place
  df = df[['Tmelt, K', 'X#1 (molar fraction)', 'Tmelt#1', 'Tmelt#2', 'MW', 'HBD_total', 'fr_Al_COO', 'fr_Ar_COO',
           'fr_Ar_N', 'fr_Ar_OH', 'fr_NH0', 'fr_NH1', 'fr_amide', 'AROM', 'ALERTS', 'n_HM', 'indexes']]
  
  df = df.rename(columns={'indexes': 'ind_syst'})
  return df
  

def encoding_DES_type(df):
  one_hot = pd.get_dummies(df['Type of DES'])
  # Concatenate the one-hot encoded columns with the original dataframe
  df = pd.concat([df, one_hot], axis=1)
  # Drop the original 'Category' column
  df = df.drop('Type of DES', axis=1)
  # Columns to move
  columns_to_move = ['1',	'2', '3', '4', '5', 'IL']
  # The column after which you need to move the columns
  target_column = 'X#1 (molar fraction)'
  # Find the index of the target column
  target_index = df.columns.get_loc(target_column)
  # Moving columns
  for col in reversed(columns_to_move):
      # Copy the column
      temp_col = df.pop(col)
      # Insert a column after the target column
      df.insert(target_index + 1, col, temp_col)
  #print(df.columns)
  return df

def split_traintest(df):
  def mix_out(x,y,groups,n_splits,test_size):
      mix_out = []
      kfold = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
      for train_idx, test_idx in kfold.split(x, y, groups):
          mix_out.append((train_idx, test_idx))
      return mix_out
  
  valid_list = []
  xy_list = []
  valid_list_s = []
  xy_list_s = []
  
  # Splitting the selection
  y = df.loc[:,'Tmelt, K']
  x = df.loc[:,'X#1 (molar fraction)':]
  mixture_out = mix_out(x, y, df['ind_syst'], 1, 0.2)
  for train_idx, val_idx in mixture_out:
      valid_list.append((x.iloc[train_idx],
                         x.iloc[val_idx],
                         y.iloc[train_idx],
                         y.iloc[val_idx]))
  
  mixture_out = mix_out(valid_list[0][0], valid_list[0][2], valid_list[0][0]['ind_syst'], 5, 0.2)
  for train_idx, val_idx in mixture_out:
      # [[x_train, x_test, y_train, y_test], [], [], [], []]
      xy_list.append((valid_list[0][0].iloc[train_idx],
                      valid_list[0][0].iloc[val_idx],
                      valid_list[0][2].iloc[train_idx],
                      valid_list[0][2].iloc[val_idx]))
  
  # Removing the 'ing_syst' column from datasets in the valid_list list
  for i, (x_train, x_val, y_train, y_val) in enumerate(valid_list):
      valid_list[i] = (x_train.drop('ind_syst', axis=1),
                       x_val.drop('ind_syst', axis=1),
                       y_train,
                       y_val)
  
  # Removing the 'ing_syst' column from datasets in the xy_list list
  for i, (x_train, x_test, y_train, y_test) in enumerate(xy_list):
      xy_list[i] = (x_train.drop('ind_syst', axis=1),
                    x_test.drop('ind_syst', axis=1),
                    y_train,
                    y_test)
  
  # Normalize the data
  scaler_x = MinMaxScaler()
  scaler_y = MinMaxScaler()
  
  # Works better with a list
  valid_list_s.append([scaler_x.fit_transform(valid_list[0][0]),
                  scaler_x.transform(valid_list[0][1]),
                  scaler_y.fit_transform(valid_list[0][2].values.reshape(-1, 1)),
                  scaler_y.transform(valid_list[0][3].values.reshape(-1, 1))])
  for fold in range(len(xy_list)):
      xy_list_s.append([scaler_x.fit_transform(xy_list[fold][0]),
                  scaler_x.transform(xy_list[fold][1]),
                  scaler_y.fit_transform(xy_list[fold][2].values.reshape(-1, 1)),
                  scaler_y.transform(xy_list[fold][3].values.reshape(-1, 1))])
  return xy_list


def compute_type(xy_list, model, types=['1', '2', '3', '4', '5', 'IL']):
    results_dict = {}
    for type in types:
        type_list = []
        for X_train, X_test, y_train, y_test in xy_list:
            type_list.append((X_train[X_train[type]==1], X_test[X_test[type]==1], y_train.loc[X_train[X_train[type]==1].index], y_test.loc[X_test[X_test[type]==1].index]))

        r2_list = []
        rmse_list = []
        r2_train_list = []
        rmse_train_list = []

        for X_train, X_test, y_train, y_test in type_list:
            # Normalize the data
            if len(X_train) > 2 and len(X_test) > 2:
                scaler_x = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
                y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()

                # Train the model
                model.fit(X_train_scaled, y_train_scaled)

                # Predict
                y_pred = model.predict(X_test_scaled)
                y1_pred = model.predict(X_train_scaled)

                # Transform the predictions back to the original scale
                y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
                y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()

                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2_train = r2_score(y_train, y1_pred)
                rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))

                r2_list.append(r2)
                rmse_list.append(rmse)
                r2_train_list.append(r2_train)
                rmse_train_list.append(rmse_train)
                
            if len(X_train) < 3:
                r2_list.append(np.nan)
                rmse_list.append(np.nan)
                r2_train_list.append(np.nan)
                rmse_train_list.append(np.nan)
                
            if len(X_train) > 2 and len(X_test) < 3:
                scaler_x = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()

                # Train the model
                model.fit(X_train_scaled, y_train_scaled)

                # Predict
                y1_pred = model.predict(X_train_scaled)

                # Transform the predictions back to the original scale
                y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()

                # Calculate metrics
                r2_train = r2_score(y_train, y1_pred)
                rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))

                r2_list.append(np.nan)
                rmse_list.append(np.nan)
                r2_train_list.append(r2_train)
                rmse_train_list.append(rmse_train)
                
        r2_list = [x if not np.isnan(x) else np.nanmean(r2_list) for x in r2_list]
        rmse_list = [x if not np.isnan(x) else np.nanmean(rmse_list) for x in rmse_list]
        r2_train_list = [x if not np.isnan(x) else np.nanmean(r2_train_list) for x in r2_train_list]
        rmse_train_list = [x if not np.isnan(x) else np.nanmean(rmse_train_list) for x in rmse_train_list]

        results_dict[type] = [
            ["R2 CV", np.nanmean(r2_list) if not np.isnan(np.nanmean(r2_list)) else 0, *r2_list],
            ["RMSE CV", np.nanmean(rmse_list) if not np.isnan(np.nanmean(rmse_list)) else 0, *rmse_list],
            ["R2 Train", np.nanmean(r2_train_list) if not np.isnan(np.nanmean(r2_train_list)) else 0, *r2_train_list],
            ["RMSE Train", np.nanmean(rmse_train_list) if not np.isnan(np.nanmean(rmse_train_list)) else 0, *rmse_train_list]
]

    return results_dict


def LR_optimization(xy_list):
  param_grid = {
      'alpha': np.arange(0.1, 100, 0.1)
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Linear Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = Ridge(**param_grid)
      # on five folds
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters', best_param)
  return best_param


def LR_best(xy_list, best_param):
  param_grid = {
      'alpha': best_param['alpha']
  }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0

  model = Ridge(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Linear Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/linear_regression{:01}.png".format(i), dpi=300)
      plt.close(fig) 

  table1 = PrettyTable()
  table1.title = "LinearRegression" # в колабе не работает, в vs code работает: It doesn't work in Colab, but it works in VS Code.
  table1.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table1.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table1.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table1.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table1.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table1)
    

  final_R.loc[len(final_R)] = ['LR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]

  return model



def DTR_optimization(xy_list):
  param_grid = {
      'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
      'max_depth': [None] + list(range(1, 20, 5)),
      'min_samples_split': np.arange(0.05, 1, 0.2),
      'min_samples_leaf': np.arange(0.05, 1, 0.2)
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Decision Tree Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = DecisionTreeRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  return


def DTR_best(xy_list, criterion='squared_error', max_depth=None, min_samples_split=0.05, min_samples_leaf=0.05):
  param_grid = {
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
  }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0

  model = DecisionTreeRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Decision Tree Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/decision_tree_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)

  table8 = PrettyTable()
  table8.title = "Decision Tree Regression" 
  table8.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table8.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table8.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table8.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table8.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table8)
  
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['DTR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]

  return model, table8, X_test_scaled

def RFR_optimization(xy_list):
  param_grid = {
      'n_estimators': [100],
      'max_depth': [None] + list(range(1, 40, 5)),
      'min_samples_split': range(10, 100, 20),
      'min_samples_leaf': range(10, 100, 20),
      'max_features': [10],
      'bootstrap': [True, False]
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print("Random Forest Regression:")
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = RandomForestRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  

def RFR_best(xy_list, n_estimators=100, max_depth=36, min_samples_split=10,
             min_samples_leaf=10, max_features=10, bootstrap=False):
  param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap
   }

  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  model = RandomForestRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Random Forest Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/random_forest_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)
      
  table2 = PrettyTable()
  table2.title = "Random Forest Regression" 
  table2.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table2.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table2.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table2.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table2.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table2)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['RFR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  return model, table2, X_test_scaled

def GBR_optimization(xy_list):
  param_grid = {
      'learning_rate': np.arange(0.001, 1, 0.3),
      'n_estimators': range(100, 400, 100),
      'min_samples_split': np.arange(0.5, 1, 0.2),
      'min_samples_leaf': np.arange(0.05, 1, 0.4),
      'max_depth': [None] + list(range(1, 8, 2))
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = GradientBoostingRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  


def GBR_best(xy_list,learning_rate=0.601,n_estimators=300,min_samples_split=0.5,min_samples_leaf=0.05,max_depth=3):
  param_grid = {
      'learning_rate': learning_rate,
      'n_estimators': n_estimators,
      'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf,
      'max_depth': max_depth
  }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = GradientBoostingRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Gradient Boosting Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/gradient_boosting_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)

  table7 = PrettyTable()
  table7.title = "Gradient Boosting Regression" 
  table7.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table7.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table7.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table7.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table7.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table7)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['GBR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]

  return model, table7, X_test_scaled
  
def CBR_optimization(xy_list):
  param_grid = {
      'learning_rate': np.arange(0.001, 0.35, 0.05),
      'depth': range(1, 9, 1),
      'l2_leaf_reg': [10],
      'iterations': [100, 200, 400],
      'border_count': [50, 150, 250],
      'bagging_temperature': [4],
      'random_strength': [4]
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Cat Boosting Regression')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = CatBoostRegressor(**param_grid, verbose=False)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  
def CBR_best(xy_list, learning_rate=0.251, depth=6, l2_leaf_reg=10,
                    iterations=100, border_count=50, bagging_temperature=4,
                    random_strength=4):
  param_grid = {
          'learning_rate': learning_rate,
          'depth': depth,
          'l2_leaf_reg': l2_leaf_reg,
          'iterations': iterations,
          'border_count': border_count,
          'bagging_temperature': bagging_temperature,
          'random_strength': random_strength
      }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = CatBoostRegressor(**param_grid, verbose=False)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Cat Boosting Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/cat_boosting_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)
  
  table4 = PrettyTable()
  table4.title = "Cat Boosting Regression" 
  table4.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table4.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table4.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table4.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table4.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table4)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['CBR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  
  return model, table4, X_test_scaled
  
def XGB_optimization(xy_list):
  param_grid = {
      'n_estimators': range(10, 200, 40),
      'max_depth': range(1, 50, 7),
      'learning_rate': np.arange(0.01, 0.25, 0.02),
      'subsample': [0.9],
      'colsample_bytree': [0.65]
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('XGBoosting Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = XGBRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  
def XGB_best(xy_list, n_estimators=170, max_depth=8, learning_rate=0.17,
             subsample=0.9, colsample_bytree=0.65):
  param_grid = {
          'n_estimators': n_estimators,
          'max_depth': max_depth,
          'learning_rate': learning_rate,
          'subsample': subsample,
          'colsample_bytree': colsample_bytree
      }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = XGBRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('XGBoosting Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/xgboosting_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)

  table3 = PrettyTable()
  table3.title = "XGBoosting Regression" 
  table3.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table3.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table3.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table3.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table3.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table3)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['XGB', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  
  return model, table3, X_test_scaled
  
  
def SVR_optimization(xy_list):
  param_grid = {
      'C': np.arange(0.1, 20, 4),
      'epsilon': np.arange(0.01, 1, 0.15),
      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
      'gamma': np.arange(0.001, 1, 0.2)
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Support Vector Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = SVR(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)

def SVR_best(xy_list, C=0.1, epsilon=0.01, kernel='rbf', gamma=0.801):
  param_grid = {
          'C': C,
          'epsilon': epsilon,
          'kernel': kernel,
          'gamma': gamma
      }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = SVR(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Support Vector Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/support_vector_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)     
  
  table5 = PrettyTable()
  table5.title = "Support Vector Regression" # в колабе не работает, в vs code работает
  table5.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table5.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table5.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table5.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table5.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table5)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['SVM', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  
  return model, table5, X_test_scaled, X_train_scaled
  

def KNN_optimization(xy_list):
  param_grid = {
      'n_neighbors': range(2, 50, 1),
      'leaf_size': [5],
      'p': [1],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('KNeighbors Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = KNeighborsRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  
def KNN_best(xy_list, n_neighbors=6, leaf_size=5, p=1, algorithm='ball_tree', metric='cityblock'):
  param_grid = {
          'n_neighbors': n_neighbors,
          'leaf_size': leaf_size,
          'p': p,
          'algorithm': algorithm,
          'metric': metric
      }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = KNeighborsRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('KNeighbors Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/k_nearest_neighbors_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)
  
  table6 = PrettyTable()
  table6.title = "KNeighbors Regression"
  table6.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table6.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table6.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table6.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table6.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table6)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['KNR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  
  return model, table6, X_test_scaled, X_train_scaled
  

def MLP_optimization(xy_list):
  param_grid = {
      'hidden_layer_sizes': [(64, 64, 64), (64, 64, 64, 64)],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'alpha': np.arange(0.0001, 1, 0.1),
      'learning_rate': ['constant', 'invscaling', 'adaptive'],
      'learning_rate_init': [0.00001, 0.01],
      'max_iter': [500]
  }
  
  # creating all iterations of hyperparameters in this range
  param_combinations = list(itertools.product(*param_grid.values()))
  param_grid_combinations = [dict(zip(param_grid.keys(), values)) for values in param_combinations]
  print('Multilayer perceptron Regression:')
  print('Number of combinations', len(param_grid_combinations))
  
  r2_list = []
  rmse_list = []
  results = []
  
  for param_grid in param_grid_combinations:
      model = MLPRegressor(**param_grid)
      # 5 Folds
      r2_list = []
      rmse_list = []
      for X_train, X_test, y_train, y_test in xy_list:
          # Normalize the data
          scaler_x = MinMaxScaler()
          scaler_y = MinMaxScaler()
          X_train_scaled = scaler_x.fit_transform(X_train)
          X_test_scaled = scaler_x.transform(X_test)
          y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
          y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
          # Train the model
          model.fit(X_train_scaled, y_train_scaled)
          # Predict
          y_pred = model.predict(X_test_scaled)
          # Transform the predictions back to the original scale
          y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
          y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
          # Calculate metrics
          r2 = r2_score(y_test, y_pred)
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          r2_list.append(r2)
          rmse_list.append(rmse)
      results.append((param_grid, np.mean(r2_list)))
  best_param, best_r2 = max(results, key=lambda x: x[1] if x[1] > 0 else 0)
  print('The best combination of hyperparameters:', best_param)
  
def MLP_best(xy_list, hidden_layer_sizes=(64, 64, 64, 64), activation='tanh',
             alpha=0.0001, learning_rate='invscaling', learning_rate_init=0.01,
             max_iter=500):
  param_grid = {
          'hidden_layer_sizes': hidden_layer_sizes,
          'activation': activation,
          'alpha': alpha,
          'learning_rate': learning_rate,
          'learning_rate_init': learning_rate_init,
          'max_iter': max_iter
      }
  
  r2_list = []
  r2_train_list = []
  rmse_list = []
  rmse_train_list = []
  results = []
  results_train = []
  i=0
  
  model = MLPRegressor(**param_grid)
  for X_train, X_test, y_train, y_test in xy_list:
      i=i+1
      # Normalize the data
      scaler_x = MinMaxScaler()
      scaler_y = MinMaxScaler()
      X_train_scaled = scaler_x.fit_transform(X_train)
      X_test_scaled = scaler_x.transform(X_test)
      y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1,1)).ravel()
      y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1,1)).ravel()
      # Train the model
      model.fit(X_train_scaled, y_train_scaled)
      # Predict
      y_pred = model.predict(X_test_scaled)
      y1_pred = model.predict(X_train_scaled)
      # Transform the predictions back to the original scale
      y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
      y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
      y1_pred = scaler_y.inverse_transform(y1_pred.reshape(-1,1)).ravel()
      # Calculate metrics
      r2 = r2_score(y_test, y_pred)
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2_train = r2_score(y_train, y1_pred)
      rmse_train = np.sqrt(mean_squared_error(y_train, y1_pred))
      r2_list.append(r2)
      rmse_list.append(rmse)
      r2_train_list.append(r2_train)
      rmse_train_list.append(rmse_train)
  
      fig, ax = plt.subplots(figsize=(15, 10))
      ax.scatter(y_train, y1_pred, color=colors[4], s=70, label='Train')
      ax.scatter(y_test, y_pred, color=colors[1], s=70, label='Cross-validation')
      ax.plot(y_train, y_train, color=colors[0])
      ax.tick_params(axis='both', which='major', labelsize=20)
      plt.xlim(100, 650)
      plt.ylim(100, 650)
      x = np.linspace(100, 650, 100)
      ax.fill_between(x, x-rmse, x+rmse, interpolate=True, color=colors[2], alpha=0.2)
      ax.set_xlabel('Test data', fontsize=35)
      ax.set_ylabel('Predicted data', fontsize=35)
      ax.set_title('Multilayer perceptron Regression', fontsize=35)
      ax.legend(fontsize = 30, frameon=False)
      plt.savefig("out/multi_layer_perceptron_regression{:01}.png".format(i), dpi=300)
      plt.close(fig)
  
  table9 = PrettyTable()
  table9.title = "Multilayer perceptron Regression"
  table9.field_names = ["metrics", "mean", "1", "2", "3", "4", "5"]
  table9.add_row(["R2 CV", np.mean(r2_list), r2_list[0], r2_list[1], r2_list[2], r2_list[3], r2_list[4]])
  table9.add_row(["RMSE CV", np.mean(rmse_list), rmse_list[0], rmse_list[1], rmse_list[2], rmse_list[3], rmse_list[4]])
  table9.add_row(["R2 Train", np.mean(r2_train_list), r2_train_list[0], r2_train_list[1], r2_train_list[2], r2_train_list[3], r2_train_list[4]])
  table9.add_row(["RMSE Train", np.mean(rmse_train_list), rmse_train_list[0], rmse_train_list[1], rmse_train_list[2], rmse_train_list[3], rmse_train_list[4]])
  print(table9)
  
  #salviamo su un dataset
  final_R.loc[len(final_R)] = ['MPR', np.mean(r2_list), np.mean(rmse_list), np.mean(r2_train_list), np.mean(rmse_train_list)]
  
  return model, table9, X_test_scaled, X_train_scaled

  

def graph_efficiencies(table2, table3, table4, table5, table6, table7, table8, table9):
  tables = [table2, table3, table4, table5, table6, table7, table8, table9]
  names = ['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP']
  
  mean_r2_train_list = []
  mean_r2_test_list = []
  std_r2_train_list = []
  std_r2_test_list = []
  
  mean_rmse_train_list = []
  mean_rmse_test_list = []
  std_rmse_train_list = []
  std_rmse_test_list = []
  
  # Cycle to extract R2_train and R2_test values from each table
  for table in tables:
      mean_r2_train = float(table._rows[2][1])
      mean_r2_test = float(table._rows[0][1])
      std_r2_train = np.std([table._rows[2][i] for i in [2, 3, 4, 5, 6]])
      std_r2_test = np.std([table._rows[0][i] for i in [2, 3, 4, 5, 6]])
  
      mean_r2_train_list.append(mean_r2_train)
      mean_r2_test_list.append(mean_r2_test)
      std_r2_train_list.append(std_r2_train)
      std_r2_test_list.append(std_r2_test)
  
      mean_rmse_train = float(table._rows[3][1])
      mean_rmse_test = float(table._rows[1][1])
      std_rmse_train = np.std([table._rows[3][i] for i in [2, 3, 4, 5, 6]])
      std_rmse_test = np.std([table._rows[1][i] for i in [2, 3, 4, 5, 6]])
  
      mean_rmse_train_list.append(mean_rmse_train)
      mean_rmse_test_list.append(mean_rmse_test)
      std_rmse_train_list.append(std_rmse_train)
      std_rmse_test_list.append(std_rmse_test)
  
  # Creating a graph
  barWidth = 0.4
  r1 = np.arange(len(names))
  r2 = [x + barWidth for x in r1]
  
  fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharex=False)
  
  ax1.bar(r1, mean_r2_train_list, color=colors[0], width=barWidth, label='R2_train', yerr=std_r2_train_list)
  ax1.bar(r2, mean_r2_test_list, color=colors[1], width=barWidth, label='R2_cross-validation', yerr=std_r2_test_list)
  ax2.bar(r1, mean_rmse_train_list, color=colors[4], width=barWidth, label='RMSE_train', yerr=std_rmse_train_list)
  ax2.bar(r2, mean_rmse_test_list, color=colors[3], width=barWidth, label='RMSE_cross-validation', yerr=std_rmse_test_list)
  
  
  ax1.grid(color='#C3C6BA', linewidth=0.3)
  ax2.grid(color='#C3C6BA', linewidth=0.3)
  
  ax1_train = mpatches.Patch(label='R2_train', color=colors[0])
  ax1_test = mpatches.Patch(label='R2_cross-val', color=colors[1])
  ax2_train = mpatches.Patch(label='RMSE_train', color=colors[4])
  ax2_test = mpatches.Patch(label='RMSE_cross-val', color=colors[3])
  
  
  
  ax1.legend(handles=[ax1_train, ax1_test], fontsize=12)
  ax2.legend(handles=[ax2_train, ax2_test],  fontsize=12)
  
  ax1.set_xticks(np.arange(8))
  ax2.set_xticks(np.arange(8))
  ax1.set_xticklabels(['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP'], fontsize=12)
  ax2.set_xticklabels(['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP'], fontsize=12)
  
  
  plt.savefig('out/graph_efficiencies.png', dpi=300, bbox_inches='tight')
  
  return
  
def graph_efficiencies_byType(rfr_type, xgb_type, cbr_type, svr_type, knn_type, gbr_type, dtr_type, mlp_type):
  keys = ['1', '2', '3', '4', '5', 'IL']
  metrics_type = [rfr_type, xgb_type, cbr_type, svr_type, knn_type, gbr_type, dtr_type, mlp_type]
  count = 0
  for key in keys:
      mean_r2_train_list = []
      mean_r2_test_list = []
      std_r2_train_list = []
      std_r2_test_list = []
  
      mean_rmse_train_list = []
      mean_rmse_test_list = []
      std_rmse_train_list = []
      std_rmse_test_list = []
  
      for metric in metrics_type:
          mean_r2_train_list.append(metric[key][2][1])
          mean_r2_test_list.append(metric[key][0][1])
          std_r2_train_list.append(np.std(metric[key][2][2:6]))
          std_r2_test_list.append(np.std(metric[key][0][2:6]))
  
          mean_rmse_train_list.append(metric[key][3][1])
          mean_rmse_test_list.append(metric[key][1][1])
          std_rmse_train_list.append(np.std(metric[key][3][2:6]))
          std_rmse_test_list.append(np.std(metric[key][1][2:6]))
  
      names = ['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP']
  
      # Creating a graph
      barWidth = 0.4
      r1 = np.arange(len(names))
      r2 = [x + barWidth for x in r1]
  
      fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(18, 3), sharex=False)
  
      ax1.bar(r1, mean_r2_train_list, color=colors[0], width=barWidth, label='R2_train', yerr=std_r2_train_list)
      ax1.bar(r2, mean_r2_test_list, color=colors[1], width=barWidth, label='R2_cross-val', yerr=std_r2_test_list)
      ax2.bar(r1, mean_rmse_train_list, color=colors[4], width=barWidth, label='RMSE_train', yerr=std_rmse_train_list)
      ax2.bar(r2, mean_rmse_test_list, color=colors[3], width=barWidth, label='RMSE_cross-val', yerr=std_rmse_test_list)
  
  
      ax1.grid(color='#C3C6BA', linewidth=0.3)
      ax2.grid(color='#C3C6BA', linewidth=0.3)
  
      ax1_train = mpatches.Patch(label='R2_train', color=colors[0])
      ax1_test = mpatches.Patch(label='R2_cross-val', color=colors[1])
      ax2_train = mpatches.Patch(label='RMSE_train', color=colors[4])
      ax2_test = mpatches.Patch(label='RMSE_cross-val', color=colors[3])
  
  
      ax1.legend(handles=[ax1_train, ax1_test], fontsize=12)
      ax2.legend(handles=[ax2_train, ax2_test], fontsize=12)
  
      ax1.set_xticks(np.arange(8))
      ax2.set_xticks(np.arange(8))
      ax1.set_xticklabels(['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP'], fontsize=12)
      ax2.set_xticklabels(['RFR', 'XGB', 'CBR', 'SVR', 'KNN', 'GBR', 'DTR', 'MLP'], fontsize=12)
  
      Names = ['Binary, Type I','Binary, Type II', 'Binary, Type III', 'Binary, Type IV', 'Binary, Type V', 'Ionic liquids']
      ax1.set_title(('R2' + ' ' + Names[count]), fontsize = 18)
      ax2.set_title(('RMSE' + ' ' + Names[count]), fontsize = 18)
      ax2.set_ylim(0, 80)
      count += 1
  
  plt.savefig('out/graph_efficiencies_byType.png', dpi=300, bbox_inches='tight')
  
  return


def save_results_csv(file_name="R2_RMSE.csv"):
  final_R.to_csv(file_name, index=False)
  print(final_R)


colors = ['#a5678e','#e8b7d4', '#beb7d9', '#7eabd4', '#31539d', 'gray'] 

plt.rcParams.update({'font.size': 16})

final_R = pd.DataFrame(columns=["Name ML",'R2 CV', 'RMSE CV', 'R2 Train', 'RMSE Train'])

