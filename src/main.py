#!/usr/bin/env python

import desmod as dm

#Dataframe formatting
df= dm.read_csv('DES_TMELT_ML.csv')

df= dm.df_fitprocessing(df)

#df_renamed = dm.column_name_NEW(df)

#Data split
xy_list= dm.split_traintest(df)

#Linear Regression
best_param= dm.LR_optimization(xy_list)

modelLR= dm.LR_best(xy_list, best_param)

#Decision Tree Regression
dm.DTR_optimization(xy_list)

modelDTR, table8, x_testDTR= dm.DTR_best(xy_list,'poisson', None, 0.05, 0.05)

#dm.feature_importance_tree('DTR', modelDTR, x_testDTR, df_renamed)


#Random Forest Regression
dm.RFR_optimization(xy_list)

modelRFR, table2, x_testRFR = dm.RFR_best(xy_list, 100, 31, 10, 10, 10, False)

#dm.feature_importance_tree('RFR', modelRFR, x_testRFR, df_renamed)

#Gradient Boosting Regression
dm.GBR_optimization(xy_list)

modelGBR, table7, x_testGBR = dm.GBR_best(xy_list, 0.301, 300, 0.5, 0.05, None)

#dm.feature_importance_tree('GBR', modelGBR, x_testGBR, df_renamed)

#Cat Boosting Regression
dm.CBR_optimization(xy_list)

modelCBR, table4, x_testCBR = dm.CBR_best(xy_list, 0.251, 7, 10, 100, 50, 4, 4)

#dm.feature_importance_tree('CBR', modelCBR, x_testCBR, df_renamed)

#Exteme Gradient Boosting
dm.XGB_optimization(xy_list)

modelXGB, table3, x_testXGB = dm.XGB_best(xy_list, 170, 15, 0.03, 0.9, 0.65)

#dm.feature_importance_tree('XGB', modelXGB, x_testXGB, df_renamed)

#Support Vector Regression
dm.SVR_optimization(xy_list)

modelSVR, table5, x_testSVR, x_testSVR  = dm.SVR_best(xy_list, 0.1, 0.01, 'rbf', 0.801)

#dm.feature_importance_kernel('SVR', modelSVR, x_testSVR, x_testSVR,  df_renamed)

#K-Nearest Neighbors Regression
dm.KNN_optimization(xy_list)

modelKNN, table6, x_testKNN, x_testKNN  = dm.KNN_best(xy_list, 4, 5, 1, 'ball_tree', 'cityblock')

#dm.feature_importance_kernel('KNN', modelKNN, x_testKNN, x_testKNN, df_renamed)

#Multilayer Perceptron
dm.MLP_optimization(xy_list)

modelMLP, table9, x_testMLP, x_testMLP  = dm.MLP_best(xy_list, (64, 64, 64), 'tanh', 0.0001, 'invscaling', 0.01, 500)

#dm.feature_importance_kernel('MLP', modelMLP, x_testMLP, x_testMLP, df_renamed)


#Saves final graphs as png
dm.graph_efficiencies(table2, table3, table4, table5, table6, table7, table8, table9)

#Saves final results of all models in a CSV file
dm.save_results_csv()
