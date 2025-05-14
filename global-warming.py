# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:54:57 2023

@author: office
"""

import pandas as pd
from prophet  import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import datetime
from datetime import datetime
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import numpy as np

dffirst = pd.read_csv('E:\old\conference borobudur\worldwide\first.csv')
dfsecond= pd.read_csv('E:\old\conference borobudur\worldwide\second.csv')
dfthird = pd.read_csv('E:\old\conference borobudur\worldwide\third.csv')


# dfsecondgab=dfsecond.merge(dfsecond2, on='Month', how='left')
# dfthirdgab=dfthird.merge(dfthird2, on='Month', how='left')


# dfsecondgab["y"]=dfsecondgab[["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"]].max(axis=1)
# dfsecondmean["y"]=dfsecondmean[["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"]].mean(axis=1)
# dfthirdgab["y"]=dfthirdgab[["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"]].max(axis=1)

#RAW
plt.plot(dfsecond['Month'],dfsecond['Yogyakarta'], label='Yogyakarta')
plt.plot(dfsecond['Month'],dfsecond['Jogja'], label='Jogja')
plt.plot(dfsecond['Month'],dfsecond['Malioboro'], label='Malioboro')
plt.plot(dfsecond['Month'],dfsecond['Borobudur'], label='Borobudur')
plt.plot(dfsecond['Month'],dfsecond['Parangtritis'], label='Parangtritis')
plt.legend()
plt.show()

#RAW
plt.plot(dfthird['Month'],dfthird['Yogyakarta'], label='Yogyakarta')
plt.plot(dfthird['Month'],dfthird['Jogja'], label='Jogja')
plt.plot(dfthird['Month'],dfthird['Malioboro'], label='Malioboro')
plt.plot(dfthird['Month'],dfthird['Borobudur'], label='Borobudur')
plt.plot(dfthird['Month'],dfthird['Parangtritis'], label='Parangtritis')
plt.plot(dfthird2['Month'],dfthird2['Yogyakarta hotel'], label='Yogyakarta hotel')
plt.plot(dfthird2['Month'],dfthird2['Jogja hotel'], label='Jogja hotel')
plt.plot(dfthird2['Month'],dfthird2['Wisata Yogyakarta'], label='Wisata Yogyakarta')
plt.plot(dfthird2['Month'],dfthird2['Wisata jogja'], label='Wisata Jogja')
plt.legend()
plt.show()


#mencari keyword terbaik untuk dibuat model second
coe_yogyakarta = np.corrcoef(dfsecond['Yogyakarta'], dffirst['total'])
coe_jogja = np.corrcoef(dfsecond['Jogja'], dffirst['total'])
coe_malioboro = np.corrcoef(dfsecond['Malioboro'], dffirst['total'])
coe_borobudur= np.corrcoef(dfsecond['Borobudur'], dffirst['total'])
coe_parangtritis= np.corrcoef(dfsecond['Parangtritis'], dffirst['total'])

coe_yogyakarta_hotel = np.corrcoef(dfsecond['Yogyakarta hotel'], dffirst['total'])
coe_jogja_hotel = np.corrcoef(dfsecond['Jogja hotel'], dffirst['total'])
coe_wisata_yogyakarta= np.corrcoef(dfsecond['Wisata Yogyakarta'], dffirst['total'])
coe_wisata_jogja= np.corrcoef(dfsecond['Wisata jogja'], dffirst['total'])

#mencari keyword terbaik untuk dibuat model third
coe_third_yogyakarta = np.corrcoef(dfthird['Yogyakarta'], dffirst['total'])
coe_third_jogja = np.corrcoef(dfthird['Jogja'], dffirst['total'])
coe_third_malioboro = np.corrcoef(dfthird['Malioboro'], dffirst['total'])
coe_third_borobudur= np.corrcoef(dfthird['Borobudur'], dffirst['total'])
coe_third_parangtritis= np.corrcoef(dfthird['Parangtritis'], dffirst['total'])

coe_third_yogyakarta_hotel = np.corrcoef(dfthird['Yogyakarta hotel'], dffirst['total'])
coe_third_jogja_hotel = np.corrcoef(dfthird['Jogja hotel'], dffirst['total'])
coe_third_wisata_yogyakarta= np.corrcoef(dfthird['Wisata Yogyakarta'], dffirst['total'])
coe_third_wisata_jogja= np.corrcoef(dfthird['Wisata jogja'], dffirst['total'])

coe_second_max=np.corrcoef(dfsecondgab['y'], dffirst['total'])
coe_second_mean=np.corrcoef(dfsecondmean['y'], dffirst['total'])
coe_third_max=np.corrcoef(dfthirdgab['y'], dffirst['total'])
coe_third_mean=np.corrcoef(dfthirdmean['y'], dffirst['total'])

#DELETE COLUMN
dfsecondyogyakarta.drop(['Jogja', 'Malioboro', 'Borobudur', 'Parangtritis'], axis=1, inplace=True)
dfsecondjogja.drop(['Yogyakarta', 'Malioboro', 'Borobudur', 'Parangtritis'], axis=1, inplace=True)
dfsecondwisatayogyakarta.drop(['Yogyakarta hotel', 'Jogja hotel', 'Wisata jogja'], axis=1, inplace=True)
dfsecondwisatajogja.drop(['Yogyakarta hotel', 'Jogja hotel', 'Wisata Yogyakarta'], axis=1, inplace=True)

dfthirdyogyakarta.drop(['Jogja', 'Malioboro', 'Borobudur', 'Parangtritis'], axis=1, inplace=True)
dfthirdjogja.drop(['Yogyakarta', 'Malioboro', 'Borobudur', 'Parangtritis'], axis=1, inplace=True)
dfthirdwisatayogyakarta.drop(['Yogyakarta hotel', 'Jogja hotel', 'Wisata jogja'], axis=1, inplace=True)
dfthirdwisatajogja.drop(['Yogyakarta hotel', 'Jogja hotel', 'Wisata Yogyakarta'], axis=1, inplace=True)

dfsecondgab.drop(["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"], axis=1, inplace=True)
dfthirdgab.drop(["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"], axis=1, inplace=True)
dfsecondmean.drop(["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"], axis=1, inplace=True)
dfthirdmean.drop(["Yogyakarta","Jogja","Malioboro","Borobudur","Parangtritis","Yogyakarta hotel","Jogja hotel","Wisata Yogyakarta","Wisata jogja"], axis=1, inplace=True)

dffirst.columns = ['ds', 'y']
dfsecondyogyakarta.columns = ['ds', 'y']
dfsecondjogja.columns = ['ds', 'y']
dfsecondwisatayogyakarta.columns = ['ds', 'y']
dfsecondwisatajogja.columns = ['ds', 'y']

dfthirdyogyakarta.columns = ['ds', 'y']
dfthirdjogja.columns = ['ds', 'y']
dfthirdwisatayogyakarta.columns = ['ds', 'y']
dfthirdwisatajogja.columns = ['ds', 'y']

dfsecondgab.columns = ['ds', 'y']
dfthirdgab.columns = ['ds', 'y']
dfsecondmean.columns = ['ds', 'y']
dfthirdmean.columns = ['ds', 'y']

dffirst['ds'] = pd.to_datetime(dffirst['ds'])
dfsecondyogyakarta['ds'] = pd.to_datetime(dfsecondyogyakarta['ds'])
dfsecondjogja['ds'] = pd.to_datetime(dfsecondjogja['ds'])
dfsecondwisatayogyakarta['ds'] = pd.to_datetime(dfsecondwisatayogyakarta['ds'])
dfsecondwisatajogja['ds'] = pd.to_datetime(dfsecondwisatajogja['ds'])
dfthirdyogyakarta['ds'] = pd.to_datetime(dfthirdyogyakarta['ds'])
dfthirdjogja['ds'] = pd.to_datetime(dfthirdjogja['ds'])
dfthirdwisatayogyakarta['ds'] = pd.to_datetime(dfthirdwisatayogyakarta['ds'])
dfthirdwisatajogja['ds'] = pd.to_datetime(dfthirdwisatajogja['ds'])

dfsecondgab['ds'] = pd.to_datetime(dfsecondgab['ds'])
dfthirdgab['ds'] = pd.to_datetime(dfthirdgab['ds'])
dfsecondmean['ds'] = pd.to_datetime(dfsecondmean['ds'])
dfthirdmean['ds'] = pd.to_datetime(dfthirdmean['ds'])

mfirst = Prophet(
    seasonality_mode='multiplicative',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=30,
    seasonality_prior_scale=35,
    mcmc_samples=0).add_seasonality('monthly', 30.5, 55).add_seasonality('yearly', 365.25, 20).fit(dffirst)
#second
msecondyogyakarta = Prophet(
    seasonality_mode='multiplicative',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=30,
    seasonality_prior_scale=35,
    mcmc_samples=0).add_seasonality('monthly', 30.5, 55).add_seasonality('yearly', 365.25, 20).fit(dfsecondyogyakarta)
#third
mthirdmean = Prophet(
    seasonality_mode='multiplicative',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=30,
    seasonality_prior_scale=35,
    mcmc_samples=0).add_seasonality('monthly', 30.5, 55).add_seasonality('yearly', 365.25, 20).fit(dfthirdmean)
futurefirst = mfirst.make_future_dataframe(periods=48, freq='MS')

futuresecondyogyakarta = msecondyogyakarta.make_future_dataframe(periods=48, freq='MS')
futuresecondjogja = msecondjogja.make_future_dataframe(periods=48, freq='MS')
futuresecondwisatayogyakarta = msecondwisatayogyakarta.make_future_dataframe(periods=48, freq='MS')
futuresecondwisatajogja = msecondwisatajogja.make_future_dataframe(periods=48, freq='MS')

futurethirdyogyakarta = mthirdyogyakarta.make_future_dataframe(periods=48, freq='MS')
futurethirdjogja = mthirdjogja.make_future_dataframe(periods=48, freq='MS')
futurethirdwisatayogyakarta = mthirdwisatayogyakarta.make_future_dataframe(periods=48, freq='MS')
futurethirdwisatajogja = mthirdwisatajogja.make_future_dataframe(periods=48, freq='MS')

futuresecondmax = msecondmax.make_future_dataframe(periods=48, freq='MS')
futurethirdmax = mthirdmax.make_future_dataframe(periods=48, freq='MS')
futuresecondmean = msecondmean.make_future_dataframe(periods=48, freq='MS')
futurethirdmean = mthirdmean.make_future_dataframe(periods=48, freq='MS')


forecastfirst = mfirst.predict(futurefirst)

forecastsecondyogyakarta = msecondyogyakarta.predict(futuresecondyogyakarta)
forecastsecondjogja = msecondjogja.predict(futuresecondjogja)
forecastsecondwisatayogyakarta = msecondwisatayogyakarta.predict(futuresecondwisatayogyakarta)
forecastsecondwisatajogja = msecondwisatajogja.predict(futuresecondwisatajogja)

forecastthirdyogyakarta = mthirdyogyakarta.predict(futurethirdyogyakarta)
forecastthirdjogja = mthirdjogja.predict(futurethirdjogja)
forecastthirdwisatayogyakarta = mthirdwisatayogyakarta.predict(futurethirdwisatayogyakarta)
forecastthirdwisatajogja = mthirdwisatajogja.predict(futurethirdwisatajogja)

forecastsecondmax= msecondmax.predict(futuresecondmax)
forecastthirdmax = mthirdmax .predict(futurethirdmax )
forecastsecondmean= msecondmean.predict(futuresecondmean)
forecastthirdmean = mthirdmean .predict(futurethirdmean)

metricfirst=forecastfirst.merge(dffirst, on='ds', how='left')
metricsecondyogyakarta=forecastsecondyogyakarta.merge(dfsecondyogyakarta, on='ds', how='left')
metricsecondjogja=forecastsecondjogja.merge(dfsecondjogja, on='ds', how='left')
metricsecondwisatayogyakarta=forecastsecondwisatayogyakarta.merge(dfsecondwisatayogyakarta, on='ds', how='left')
metricsecondwisatajogja=forecastsecondwisatajogja.merge(dfsecondwisatajogja, on='ds', how='left')

metricthirdyogyakarta=forecastthirdyogyakarta.merge(dfthirdyogyakarta, on='ds', how='left')
metricthirdjogja=forecastthirdjogja.merge(dfthirdjogja, on='ds', how='left')
metricthirdwisatayogyakarta=forecastthirdwisatayogyakarta.merge(dfthirdwisatayogyakarta, on='ds', how='left')
metricthirdwisatajogja=forecastthirdwisatajogja.merge(dfthirdwisatajogja, on='ds', how='left')

metricsecondmax=forecastsecondmax.merge(dfsecondgab, on='ds', how='left')
metricthirdmax=forecastthirdmax.merge(dfthirdgab, on='ds', how='left')
metricsecondmean=forecastsecondmean.merge(dfsecondmean, on='ds', how='left')
metricthirdmean=forecastthirdmean.merge(dfthirdmean, on='ds', how='left')

#PREDICTION
plt.plot(metricfirst.ds, metricfirst.y,label='first Visitor')
plt.plot(metricfirst.ds, metricfirst.yhat,label='Prediction first Visitor')
plt.legend()
plt.show()

plt.plot(metricsecondyogyakarta.ds,metricsecondyogyakarta.y, label= 'Yogyakarta')
plt.plot(metricsecondyogyakarta.ds,metricsecondyogyakarta.yhat, label='Yogyakarta Prediction')
plt.legend()
plt.show()

plt.plot(metricsecondjogja.ds,metricsecondjogja.y, label= 'Jogja')
plt.plot(metricsecondjogja.ds,metricsecondjogja.yhat, label='Jogja Prediction')
plt.legend()
plt.show()

plt.plot(metricsecondwisatayogyakarta.ds,metricsecondwisatayogyakarta.y, label= 'Wisata Yogyakarta')
plt.plot(metricsecondwisatayogyakarta.ds,metricsecondwisatayogyakarta.yhat, label='Wisata Yogyakarta Prediction')
plt.legend()
plt.show()

plt.plot(metricsecondwisatajogja.ds,metricsecondwisatajogja.y, label= 'Wisata Jogja')
plt.plot(metricsecondwisatajogja.ds,metricsecondwisatajogja.yhat, label='Wisata Jogja Prediction')
plt.legend()
plt.show()

##third
plt.plot(metricthirdyogyakarta.ds,metricthirdyogyakarta.y, label= 'Yogyakarta')
plt.plot(metricthirdyogyakarta.ds,metricthirdyogyakarta.yhat, label='Yogyakarta Prediction')
plt.legend()
plt.show()

plt.plot(metricthirdjogja.ds,metricthirdjogja.y, label= 'Jogja')
plt.plot(metricthirdjogja.ds,metricthirdjogja.yhat, label='Jogja Prediction')
plt.legend()
plt.show()

plt.plot(metricthirdwisatayogyakarta.ds,metricthirdwisatayogyakarta.y, label= 'Wisata Yogyakarta')
plt.plot(metricthirdwisatayogyakarta.ds,metricthirdwisatayogyakarta.yhat, label='Wisata Yogyakarta Prediction')
plt.legend()
plt.show()

plt.plot(metricthirdwisatajogja.ds,metricthirdwisatajogja.y, label= 'Wisata Jogja')
plt.plot(metricthirdwisatajogja.ds,metricthirdwisatajogja.yhat, label='Wisata Jogja Prediction')
plt.legend()
plt.show()

#MAX
plt.plot(metricsecondmax.ds,metricsecondmax.y, label= 'second Max')
plt.plot(metricsecondmax.ds,metricsecondmax.yhat, label='second Max Prediction')
plt.plot(metricthirdmax.ds,metricthirdmax.y, label= 'third Max')
plt.plot(metricthirdmax.ds,metricthirdmax.yhat, label='third Max Prediction')
plt.legend()
plt.show()


#Mean
plt.plot(metricsecondmean.ds,metricsecondmean.y, label= 'second Mean')
plt.plot(metricsecondmean.ds,metricsecondmean.yhat, label='second Mean Prediction')
plt.plot(metricthirdmean.ds,metricthirdmean.y, label= 'third Mean')
plt.plot(metricthirdmean.ds,metricthirdmean.yhat, label='third Mean Prediction')
plt.legend()
plt.show()

metric2first=metricfirst
metric2secondyogyakarta=metricsecondyogyakarta
metric2secondjogja=metricsecondjogja 
metric2secondwisatayogyakarta=metricsecondwisatayogyakarta
metric2secondwisatajogja=metricsecondwisatajogja
metric2thirdyogyakarta=metricthirdyogyakarta
metric2thirdjogja=metricthirdjogja
metric2thirdwisatayogyakarta=metricthirdwisatayogyakarta
metric2thirdwisatajogja=metricthirdwisatajogja

metric2secondmax=metricsecondmax
metric2thirdmax=metricthirdmax
metric2secondmean=metricsecondmean
metric2thirdmean=metricthirdmean

# UBAH KEMBALI ISI METRIC DENGAN TAHUN 2020-2024
# metric3first=metricfirst
# metric3secondyogyakarta=metricsecondyogyakarta
# metric3secondjogja=metricsecondjogja 
# metric3secondyogyakarta2=metricsecondyogyakarta2
# metric3secondjogja2=metricsecondjogja2
# metric3thirdyogyakarta=metricthirdyogyakarta
# metric3thirdjogja=metricthirdjogja
# metric3thirdyogyakarta2=metricthirdyogyakarta2
# metric3thirdjogja2=metricthirdjogja2

metric2first.dropna(inplace=True) 
metric2secondyogyakarta.dropna(inplace=True) 
metric2secondjogja.dropna(inplace=True) 
metric2secondwisatayogyakarta.dropna(inplace=True) 
metric2secondwisatajogja.dropna(inplace=True) 

metric2thirdyogyakarta.dropna(inplace=True) 
metric2thirdjogja.dropna(inplace=True) 
metric2thirdwisatayogyakarta.dropna(inplace=True) 
metric2thirdwisatajogja.dropna(inplace=True) 

metricsecondmax.dropna(inplace=True) 
metricthirdmax.dropna(inplace=True) 
metricsecondmean.dropna(inplace=True) 
metricthirdmean.dropna(inplace=True) 


meanfirst=mean_absolute_percentage_error(metric2first.y, metric2first.yhat)
meansecondyogyakarta=mean_absolute_percentage_error(metric2secondyogyakarta.y, metric2secondyogyakarta.yhat)
meansecondjogja=mean_absolute_percentage_error(metric2secondjogja.y, metric2secondjogja.yhat)
meansecondwisatayogyakarta=mean_absolute_percentage_error(metric2secondwisatayogyakarta.y, metric2secondwisatayogyakarta.yhat)
meansecondwisatajogja=mean_absolute_percentage_error(metric2secondwisatajogja.y, metric2secondwisatajogja.yhat)
meanthirdyogyakarta=mean_absolute_percentage_error(metric2thirdyogyakarta.y, metric2thirdyogyakarta.yhat)
meanthirdjogja=mean_absolute_percentage_error(metric2thirdjogja.y, metric2thirdjogja.yhat)
meanthirdwisatayogyakarta=mean_absolute_percentage_error(metric2thirdwisatayogyakarta.y, metric2thirdwisatayogyakarta.yhat)
meanthirdwisatajogja=mean_absolute_percentage_error(metric2thirdwisatajogja.y, metric2thirdwisatajogja.yhat)

meansecondmax=mean_absolute_percentage_error(metric2secondmax.y, metric2secondmax.yhat)
meanthirdmax=mean_absolute_percentage_error(metricthirdmax.y, metricthirdmax.yhat)

meansecondmean=mean_absolute_percentage_error(metric2secondmean.y, metric2secondmean.yhat)
meanthirdmean=mean_absolute_percentage_error(metricthirdmean.y, metricthirdmean.yhat)

r2_scorefirst=r2_score(metricfirst.y, metricfirst.yhat)

r2_scoresecondyogyakarta=r2_score(metricsecondyogyakarta.y, metricsecondyogyakarta.yhat)
r2_scoresecondjogja=r2_score(metricsecondjogja.y, metricsecondjogja.yhat)
r2_scoresecondwisatayogyakarta=r2_score(metricsecondwisatayogyakarta.y, metricsecondwisatayogyakarta.yhat)
r2_scoresecondwisatajogja=r2_score(metricsecondwisatajogja.y, metricsecondwisatajogja.yhat)

r2_scorethirdyogyakarta=r2_score(metricthirdyogyakarta.y, metricthirdyogyakarta.yhat)
r2_scorethirdjogja=r2_score(metricthirdjogja.y, metricthirdjogja.yhat)
r2_scorethirdwisatayogyakarta=r2_score(metricthirdwisatayogyakarta.y, metricthirdwisatayogyakarta.yhat)
r2_scorethirdwisatajogja=r2_score(metricthirdwisatajogja.y, metricthirdwisatajogja.yhat)

r2_scoresecondmax=r2_score(metricsecondmax.y, metricsecondmax.yhat)
r2_scorethirdmax=r2_score(metricthirdmax.y, metricthirdmax.yhat)
r2_scoresecondmean=r2_score(metricsecondmean.y, metricsecondmean.yhat)
r2_scorethirdmean=r2_score(metricthirdmean.y, metricthirdmean.yhat)

#CROSS VALIDATION first
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dffirst_cv = cross_validation(mfirst,initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dffirst_cv.head()
dffirst_p = performance_metrics(dffirst_cv)
dffirst_p.head()
fig = plot_cross_validation_metric(dffirst_cv, metric='mape')


#CROSS VALIDATION second YOGYAKARTA
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondyogyakarta_cv = cross_validation(msecondyogyakarta,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondyogyakarta_cv.head()
dfsecondyogyakarta_p = performance_metrics(dfsecondyogyakarta_cv)
dfsecondyogyakarta_p.head()
fig2 = plot_cross_validation_metric(dfsecondyogyakarta_cv, metric='mape')

#CROSS VALIDATION second JOGJA
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondjogja_cv = cross_validation(msecondjogja,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondjogja_cv.head()
dfsecondjogja_p = performance_metrics(dfsecondjogja_cv)
dfsecondjogja_p.head()
fig2 = plot_cross_validation_metric(dfsecondjogja_cv, metric='mape')

#CROSS VALIDATION second
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondwisatayogyakarta_cv = cross_validation(msecondwisatayogyakarta,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondwisatayogyakarta_cv.head()
dfsecondwisatayogyakarta_p = performance_metrics(dfsecondwisatayogyakarta_cv)
dfsecondwisatayogyakarta_p.head()
fig3 = plot_cross_validation_metric(dfsecondwisatayogyakarta_cv, metric='mape')

#CROSS VALIDATION second
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondwisatajogja_cv = cross_validation(msecondwisatajogja,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondwisatajogja_cv.head()
dfsecondwisatajogja_p= performance_metrics(dfsecondwisatajogja_cv)
dfsecondwisatajogja_p.head()
fig3 = plot_cross_validation_metric(dfsecondwisatajogja_cv, metric='mape')

#CROSS VALIDATION second MAX
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondmax_cv = cross_validation(msecondmax,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondmax_cv.head()
dfsecondmax_p = performance_metrics(dfsecondmax_cv)
dfsecondmax_p.head()
fig3 = plot_cross_validation_metric(dfsecondmax_cv, metric='mape')

#CROSS VALIDATION second MEAN
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfsecondmean_cv = cross_validation(msecondmean,initial='1825 days',  period=730, cutoffs=cutoffs, horizon='365 days')
dfsecondmean_cv.head()
dfsecondmean_p= performance_metrics(dfsecondmean_cv)
dfsecondmean_p.head()
fig3 = plot_cross_validation_metric(dfsecondmean_cv, metric='mape')

#CROSS VALIDATION third
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdyogyakarta_cv = cross_validation(mthirdyogyakarta, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdyogyakarta_cv.head()
dfthirdyogyakarta_p = performance_metrics(dfthirdyogyakarta_cv)
dfthirdyogyakarta_p.head()
fig4 = plot_cross_validation_metric(dfthirdyogyakarta_cv, metric='mape')

#CROSS VALIDATION third
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdjogja_cv = cross_validation(mthirdjogja, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdjogja_cv.head()
dfthirdjogja_p = performance_metrics(dfthirdjogja_cv)
dfthirdjogja_p.head()
fig4 = plot_cross_validation_metric(dfthirdjogja_cv, metric='mape')

#CROSS VALIDATION third
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdwisatayogyakarta_cv = cross_validation(mthirdwisatayogyakarta, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdwisatayogyakarta_cv.head()
dfthirdwisatayogyakarta_p = performance_metrics(dfthirdwisatayogyakarta_cv)
dfthirdwisatayogyakarta_p.head()
fig5 = plot_cross_validation_metric(dfthirdwisatayogyakarta_cv, metric='mape')

#CROSS VALIDATION third
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdwisatajogja_cv = cross_validation(mthirdwisatajogja, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdwisatajogja_cv.head()
dfthirdwisatajogja_p = performance_metrics(dfthirdwisatajogja_cv)
dfthirdwisatajogja_p.head()
fig5 = plot_cross_validation_metric(dfthirdwisatajogja_cv, metric='mape')

#CROSS VALIDATION third MAX
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdmax_cv = cross_validation(mthirdmax, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdmax_cv.head()
dfthirdmax_p = performance_metrics(dfthirdmax_cv)
dfthirdmax_p.head()
fig5 = plot_cross_validation_metric(dfthirdmax_cv, metric='mape')

#CROSS VALIDATION third MEAN
cutoffs = pd.date_range(start='2018-01-01', end='2018-12-01', freq='6MS')
dfthirdmean_cv = cross_validation(mthirdmean, initial='1825 days', period=730, cutoffs=cutoffs, horizon='365 days')
dfthirdmean_cv.head()
dfthirdmean_p = performance_metrics(dfthirdmean_cv)
dfthirdmean_p.head()
fig5 = plot_cross_validation_metric(dfthirdmean_cv, metric='mape')

#CROSS VALIDATION of 3 MAPE
#plt.plot(dffirst_p.horizon, dffirst_p.mape, label='first Visitor')
plt.plot(dfsecondyogyakarta_p.horizon, dfsecondyogyakarta_p.mape, label='second Yogyakarta')
plt.plot(dfsecondjogja_p.horizon, dfsecondjogja_p.mape, label='second Jogja')
plt.plot(dfsecondwisatayogyakarta_p.horizon, dfsecondwisatayogyakarta_p.mape, label= 'second Wisata Yogyakarta')
plt.plot(dfsecondwisatajogja_p.horizon, dfsecondwisatajogja_p.mape, label='second Wisata Jogja')
plt.plot(dfsecondmax_p.horizon, dfsecondmax_p.mape, label='second Max')
plt.plot(dfsecondmean_p.horizon, dfsecondmean_p.mape, label='second Mean')
plt.legend()
plt.show()

#plt.plot(dffirst_p.horizon, dffirst_p.mape, label='first Visitor')
plt.plot(dfthirdyogyakarta_p.horizon, dfthirdyogyakarta_p.mape, label='third Yogyakarta')
plt.plot(dfthirdjogja_p.horizon, dfthirdjogja_p.mape, label='third Jogja')
plt.plot(dfthirdwisatayogyakarta_p.horizon, dfthirdwisatayogyakarta_p.mape, label= 'third Wisata Yogyakarta')
plt.plot(dfthirdwisatajogja_p.horizon, dfthirdwisatajogja_p.mape, label='third Wisata Jogja')
plt.plot(dfthirdmax_p.horizon, dfthirdmax_p.mape, label='third Max')
plt.plot(dfthirdmean_p.horizon, dfthirdmean_p.mape, label='third Mean')
plt.legend()
plt.show()

# #CROSS VALIDATION of 3 MAE
# plt.plot(dffirst_p.horizon, dffirst_p.mae, label='MAPE score - first Visitor')
# plt.plot(dfsecondyogyakarta_p.horizon, dfsecondyogyakarta_p.mae, label='MAPE score - second Yogyakarta')
# plt.plot(dfsecondjogja_p.horizon, dfsecondjogja_p.mae, label='MAPE score - second Jogja')
# plt.plot(dfsecond2_p.horizon, dfsecond2_p.mae, label='MAPE score - second Wisata Jogja')
# plt.plot(dfthird_p.horizon, dfthird_p.mape, label='MAPE score - third Yogyakarta')
# plt.plot(dfthirdjogja_p.horizon, dfthirdjogja_p.mae, label='MAPE score - third Jogja')
# plt.plot(dfthird2_p.horizon, dfthird2_p.mae, label='MAPE score - third Wisata Jogja')
# plt.legend()
# plt.show()


#PREDICTION
plt.plot(metricfirst.ds, metricfirst.y,label='first Visitor')
plt.plot(metricfirst.ds, metricfirst.yhat,label='Prediction first Visitor')
plt.legend()
plt.show()

plt.plot(metricsecondyogyakarta.ds,metricsecondyogyakarta.y, label= 'second Yogyakarta')
plt.plot(metricsecondyogyakarta.ds,metricsecondyogyakarta.yhat, label='Prediction ')
plt.legend()
plt.show()

plt.plot(metricsecondjogja.ds,metricsecondjogja.y, label= 'second Jogja')
plt.plot(metricsecondjogja.ds,metricsecondjogja.yhat, label='Prediction')
plt.legend()
plt.show()

plt.plot(metricthirdyogyakarta.ds,metricthirdyogyakarta.y,  label='third Yogyakarta')
plt.plot(metricthirdyogyakarta.ds,metricthirdyogyakarta.yhat,  label='Prediction')
plt.legend()
plt.show()

plt.plot(metricthirdjogja.ds,metricthirdjogja.y,  label='third Wisata Jogja')
plt.plot(metricthirdjogja.ds,metricthirdjogja.yhat,  label='Prediction')
plt.legend()
plt.show()

#MAX
plt.plot(metricsecondmax.ds,metricsecondmax.y,  label='second Max')
plt.plot(metricsecondmax.ds,metricsecondmax.yhat,  label='Prediction')
plt.legend()
plt.show()

#MAX
plt.plot(metricthirdmax.ds,metricthirdmax.y,  label='third Max')
plt.plot(metricthirdmax.ds,metricthirdmax.yhat,  label='Prediction')
plt.legend()
plt.show()

#MEAN
plt.plot(metricsecondmean.ds,metricsecondmean.y,  label='second Mean')
plt.plot(metricsecondmean.ds,metricsecondmean.yhat,  label='Prediction')
plt.legend()
plt.show()

#MEAN
plt.plot(metricwebmean.ds,metricwebmean.y,  label='Web Mean')
plt.plot(metricwebmean.ds,metricwebmean.yhat,  label='Prediction')
plt.legend()
plt.show()

mfirst.plot_components(forecastfirst)
mwebjogja.plot_components(forecastwebjogja)
msecondjogja.plot_components(forecastsecondjogja)

msecondmax.plot_components(forecastsecondmax)
mwebmax.plot_components(forecastwebmax)
msecondmean.plot_components(forecastsecondmean)
mthirdmean.plot_components(forecastthirdmean)