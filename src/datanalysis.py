#!/usr/bin/env python

import desmod as dm

df= dm.read_csv('DES_TMELT_V.csv')

df= dm.df_foranalysis(df)

dm.popular_components(df)

dm.melting_temperature_distribution(df)

