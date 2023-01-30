

import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
#import json

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy import stats
from statsmodels.tsa.api import SARIMAX

import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return dt.date(year, month, day)

def tsplot(y, lags=None, figsize=(18, 16), style='bmh',target='Live births',show=True,takeSquare=False):
    if takeSquare: y = np.square(y); title = 'Time Series Analysis Plots of Square of '+ target
    else: title = 'Time Series Analysis Plots of '+ target
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        stats.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
    if show: plt.show()


def showDTplot(df,col2show=['realbirths','yhat','yhat_lower', 'yhat_upper'],offset=2,title='Live Births in Sweden',ylabel='Births per Month',confInt=[],target='Live births',horizon=12,year=False,showBorder=False):
    colors = ['b','g','r','yellow','black']
    fig, axes = plt.subplots(1,1,sharex=True,figsize=(16,10))
    fig.autofmt_xdate()
    axes.set_title(title,fontsize=14)
    axes.set_ylabel(ylabel, fontsize=12)
    if year: axes.set_xlabel("Year", fontsize=12)
    else: axes.set_xlabel("Time in Month-Year", fontsize=12)
    if showBorder: axes.axvline(x=df.index[len(df)-horizon-1],linewidth=1, color='r',linestyle='--',label='end of real data')
    if confInt != []:
        for i,ele in enumerate(confInt):
            conf_label = ele[0] + '-' + ele[1]
            axes.fill_between(df.index[offset:], df[ele[0]].values[offset:], df[ele[1]].values[offset:],color=colors[i], alpha=0.1)
    for ele in col2show:
        if ele == target: axes.plot(df.index[offset:-horizon], df[ele][offset:-horizon],label=ele)
        else: axes.plot(df.index[offset:], df[ele][offset:],label=ele)
    if year: xfmt = mdates.DateFormatter('%Y')
    else: xfmt = mdates.DateFormatter('%m-%y')
    axes.xaxis.set_major_formatter(xfmt)
    plt.legend()
    fig.tight_layout()
    plt.show()



# your path to documents
save_path = 'YOUR_PATH'



births_month = pd.read_csv(save_path + 'births_month3.csv',delimiter=',') 

births_month.head()
births_month.tail()

births_month.columns
"""
Index(['Year', 'Month', 'Population at the end of period',
       'Population growth1', 'Live births', 'Still births', 'Deaths',
       'Population surplus', 'Immigrants', 'Emigrants',
       'Surplus of immigrants', 'Date'],
      dtype='object')"""


# check if data contains nan
births_month.isnull().sum()
"""
Year                               0
Month                              0
Population at the end of period    0
Population growth1                 0
Live births                        0
Still births                       0
Deaths                             0
Population surplus                 0
Immigrants                         0
Emigrants                          0
Surplus of immigrants              0
Date                               0
dtype: int64"""



dates = [dt.datetime.strptime(x, '%Y-%m-%d').date() for x in births_month['Date'].values]

# set dataset index as datetime objects
births_month.index = dates

# show a chart 
showDTplot(births_month,col2show=['Live births'],offset=0,title='Live Births in Sweden',ylabel='Births per Month',target='')


#showDTplot(births_month,col2show=['Surplus of immigrants'],offset=0,title='Surplus of immigrants in Sweden',ylabel='per Month')




### prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

horizon = 12
df = births_month[['Date','Live births']].copy()
df.columns = ['ds','y'] 
df['ds'] = pd.to_datetime(df['ds'])
df['y']=df['y'].astype(float)
m = Prophet(changepoint_prior_scale=0.03, daily_seasonality=False,mcmc_samples=100,interval_width=0.75,changepoint_range=0.95,yearly_seasonality=6,weekly_seasonality=False,seasonality_prior_scale= 0.1)
m.fit(df)
future = m.make_future_dataframe(periods = horizon) # , freq = 1
forecast = m.predict(future)

fig1 = m.plot(forecast);plt.show()
fig2 = m.plot_components(forecast);plt.show()



# validate model
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_p = performance_metrics(df_cv)
fig = plot_cross_validation_metric(df_cv, metric = 'mape');plt.show()

#save results
forecast.to_csv(save_path+'births_month_forecast.csv', index=False,header=True)

# load results
#forecast = pd.read_csv(save_path+'births_month_forecast.csv',delimiter=',') 


# convert dates to datetime objects and set dataset index with it
dates = [dt.datetime.strptime(x, '%Y-%m-%d').date() for x in births_month['Date'].values]

lastdate = dates[-1]
for i in range(1,horizon+1,1):
    dates.append(add_months(lastdate, i))

datesstr = [x.isoformat() for x in dates]


forecast['Date'] = datesstr
forecast.index = dates


# inculde true data in forecast dataset, forecast period (horizon) is set to a constant value
realbirths = births_month['Live births'].values.tolist()
realbirths = realbirths + [realbirths[-1]] * horizon
forecast['Live births'] = realbirths


showDTplot(forecast,col2show=['Live births','yhat','trend'],offset=0,title='Live Births in Sweden - prediction for next 12 months',ylabel='Births per Month',confInt=[('yhat_lower','yhat_upper'),('trend_lower', 'trend_upper')],showBorder=True)


#dt.date(2022,12,31) > forecast.index > dt.date(2022,1,1)



forecast['cycle'] = forecast['Live births'] - forecast['trend']

plt.plot(forecast['cycle'])
plt.show()

#####################333

X = forecast['cycle'].values[:-horizon]
X.shape


tsplot(y=X, lags=None, figsize=(18, 16), style='bmh',target='Live births Cycle only',show=True,takeSquare=True)




sarimax_mod = SARIMAX(X, order=((2,12), 0, 1), trend="ct")
sarimax_res = sarimax_mod.fit()
print(sarimax_res.summary())

"""
                                 SARIMAX Results                                  
==================================================================================
Dep. Variable:                          y   No. Observations:                   47
Model:             SARIMAX([2, 12], 0, 1)   Log Likelihood                -332.109
Date:                    Mon, 30 Jan 2023   AIC                            676.217
Time:                            13:10:20   BIC                            687.318
Sample:                                 0   HQIC                           680.395
                                     - 47                                         
Covariance Type:                      opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.9866     18.104      0.054      0.957     -34.497      36.471
drift         -0.8220      2.067     -0.398      0.691      -4.874       3.230
ar.L2          0.0648      0.042      1.541      0.123      -0.018       0.147
ar.L12         0.8999      0.034     26.330      0.000       0.833       0.967
ma.L1          0.5001      0.139      3.609      0.000       0.229       0.772
sigma2      5.075e+04   1.14e+04      4.463      0.000    2.85e+04     7.3e+04
===================================================================================
Ljung-Box (L1) (Q):                   0.15   Jarque-Bera (JB):                 1.71
Prob(Q):                              0.70   Prob(JB):                         0.43
Heteroskedasticity (H):               0.79   Skew:                            -0.21
Prob(H) (two-sided):                  0.64   Kurtosis:                         2.17
===================================================================================
"""


from statsmodels.tsa.holtwinters import ExponentialSmoothing

n_test = 12
y = forecast['trend'].values
test = y[-n_test:]
train = y[:-n_test]

model = ExponentialSmoothing(train, trend='add', damped_trend=True, seasonal=None) #  seasonal_periods=p
model_fit = model.fit(optimized=True) # , use_boxcox=b, remove_bias=r
yhat = model_fit.predict(len(train),len(train)+n_test-1)

plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.plot(np.concatenate((model_fit.fittedvalues,yhat), axis=None),label="Prediction")
plt.plot(np.concatenate((train,test), axis=None),label='Real Trend')
plt.legend()
plt.show()

forecast['trend exp smoothing'] = np.concatenate((model_fit.fittedvalues,yhat), axis=None)


forecast['sarimax mean expSmooth'] = forecast['trend exp smoothing'].values + np.concatenate((sarimax_res.fittedvalues, predict.predicted_mean), axis=0)
forecast['sarimax lower expSmooth'] = forecast['trend exp smoothing'].values + np.concatenate((sarimax_res.fittedvalues, predict_ci[:,0]), axis=0)
forecast['sarimax upper expSmooth'] = forecast['trend exp smoothing'].values + np.concatenate((sarimax_res.fittedvalues, predict_ci[:,1]), axis=0)



showDTplot(forecast,col2show=['Live births','sarimax mean expSmooth','trend exp smoothing'],offset=0,title='Live Births in Sweden',ylabel='Births per Month',confInt=[('sarimax lower expSmooth','sarimax upper expSmooth')],showBorder=True,horizon=12)


forecast.to_csv(save_path+'births_month_forecast2.csv', index=False,header=True)





"""


orders = [(12),(2,12),(2,10,12),(2,10,12),(1,12),(2,8,12),(2,3,12)]

res_list = []

for o in orders:
    try:
        sarimax_mod = SARIMAX(X, order=(o, 0, 0), trend="c")
        sarimax_res = sarimax_mod.fit()
        res_list.append([o,sarimax_res.llf,sarimax_res.aic])
    except Exception as e:
        pass


aics = np.array([x[2] for x in res_list])
aics # shows: array([685.04702963, 682.78285859, 684.34102135, 684.34102135, 678.04835428,  10.        ])
np.argmax(aics == aics.min()) # shows: 5
best_order = res_list[5][0] # (2, 3, 12)



llf = np.array([x[1] for x in res_list])
llf # array([-328.52351481, -337.3914293 , -337.17051067, -337.17051067,-335.02417714,    0.        ])

np.argmax(llf == llf.min()) # shows: 1
best_order = res_list[1][0] # (2, 12)

"""



tsplot(y=sarimax_res.resid, lags=None, figsize=(18, 16), style='bmh',target='sarimax_res.resid',show=True,takeSquare=True)


predict = sarimax_res.get_prediction(start=len(X), end=len(X)+horizon-1, dynamic=False)
predict_ci = predict.conf_int()



forecast['sarimax mean'] = forecast['trend'].values + np.concatenate((sarimax_res.fittedvalues, predict.predicted_mean), axis=0)
forecast['sarimax lower'] = forecast['trend'].values + np.concatenate((sarimax_res.fittedvalues, predict_ci[:,0]), axis=0)
forecast['sarimax upper'] = forecast['trend'].values + np.concatenate((sarimax_res.fittedvalues, predict_ci[:,1]), axis=0)


showDTplot(forecast,col2show=['Live births','sarimax mean'],offset=0,title='Live Births in Sweden',ylabel='Births per Month',confInt=[('sarimax lower','sarimax upper')],showBorder=True,horizon=12)



forecast = forecast.astype({"sarimax mean": int, "sarimax lower": int,"sarimax upper": int})

forecast = forecast.astype({"sarimax mean expSmooth": int, "sarimax lower expSmooth": int,"sarimax upper expSmooth": int})


mask = (forecast.index >= dt.date(2022,12,1)) & (forecast.index <= dt.date(2023,11,30))
forecastvalues = forecast[["Date","sarimax lower expSmooth","sarimax mean expSmooth","sarimax upper expSmooth"]].loc[mask].values # 7350

"""
forecastvalues

Year-Month  lower   mean    upper
'2022-12', 6836, 7277, 7719,
'2023-01', 7469, 7963, 8456,
'2023-02', 7113, 7607, 8102,
'2023-03', 7728, 8223, 8718,
'2023-04', 7363, 7857, 8352,
'2023-05', 8087, 8581, 9076,
'2023-06', 8018, 8512, 9007,
'2023-07', 7793, 8288, 8783,
'2023-08', 7911, 8406, 8901,
'2023-09', 7002, 7497, 7992,
'2023-10', 6804, 7299, 7793,
'2023-11', 6167, 6662, 7157
"""




# get forecast for dec 2022, not yet published
mask = (forecast.index >= dt.date(2022,12,1)) & (forecast.index <= dt.date(2022,12,31))
dec22forecast = forecast["sarimax mean expSmooth"].loc[mask].values # 7277


mask = (births_month.index >= dt.date(2022,1,1)) & (births_month.index <= dt.date(2022,12,31))
births22 = int(births_month['Live births'].loc[mask].values.sum() ) 
births22 # jan-nov

births22 += 7719 #upper forecast for dec 2022,  2022 total: 104930


########### births per year ######

births = pd.read_csv(save_path + 'be0101_tab9utv1749-2021-1a.csv',delimiter=',') 
birthsPerYear = births['Live births'].values.tolist() + [births22]
years = births['year'].values.tolist() + [2022]
births1749_2022 = pd.DataFrame()
births1749_2022['year'] = years
births1749_2022['Live births'] = birthsPerYear
dates = [dt.date(x,12,31) for x in years]
datesstr = [x.isoformat() for x in dates]

births1749_2022['Date'] = datesstr 
births1749_2022.index = dates

showDTplot(births1749_2022,col2show=['Live births'],offset=0,title='Live Births in Sweden',ylabel='Births per Year',confInt=[],target='',year=True)


birth_proc = births1749_2022['Live births'].pct_change().fillna(0.)  #.dropna()
births1749_2022['Live births pct'] = birth_proc.values

showDTplot(births1749_2022,col2show=['Live births pct'],offset=0,title='Live Births in Sweden in procent change from previous year',ylabel='Births per Year 0.1 = 10%',confInt=[],target='',year=True)

showDTplot(births1749_2022,col2show=['Live births pct'],offset=174,title='Live Births in Sweden last 100 years in procent change from previous year',ylabel='Births per Year 0.1 = 10%',confInt=[],target='',year=True)

births1749_2022['Live births pct stand'] = (births1749_2022['Live births pct'].values - births1749_2022['Live births pct'].mean())/births1749_2022['Live births pct'].std()
showDTplot(births1749_2022,col2show=['Live births pct stand'],offset=174,title='Live Births in Sweden last 100 years, standarized change from previous year',ylabel='std',confInt=[],target='',year=True)

showDTplot(births1749_2022,col2show=['Live births pct stand'],offset=0,title='Live Births in Sweden last 273 years, standarized change from previous year',ylabel='std',confInt=[],target='',year=True)



births1749_2022.to_csv(save_path+'births_per_year_1749_2022.csv',index=False,header=True,sep=',')


births1749_2022.iloc[174]
"""year                           1923
Live births                  113435
Date                     1923-12-31
Live births pct           -0.030022
Live births pct stand     -0.639977
Name: 1923-12-31, dtype: object"""

births1749_2022.iloc[-1]
"""year                           2022
Live births                  104930
Date                     2022-12-31
Live births pct            -0.08168
Live births pct stand     -1.629892
Name: 2022-12-31, dtype: object"""



#### fit data to t-distribution. calculate probability for event
stats.t.fit(births1749_2022['Live births pct'].values, floc=births1749_2022['Live births pct'].mean(), fscale=births1749_2022['Live births pct'].std())
# (9.732714843750019, 0.0033740561159216614, 0.052183843417941314)

par = {'df':9.732714843750019, 'loc':0.0033740561159216614,'scale':0.052183843417941314}

fig, ax = plt.subplots(1, 1)
plt.title('Student’s t Distribution of Swedens yearly change in births df=9.73,loc=0.003,scale=0.052')
x = np.linspace(stats.t.ppf(0.01, **par),stats.t.ppf(0.99, **par), len(births1749_2022))

ax.plot(x, stats.t.pdf(x, **par),'r-', lw=5, alpha=0.6, label='t pdf')

rv = stats.t(**par)

ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#ax.hist(births1749_2022['Live births pct'].values, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.hist(births1749_2022['Live births pct'].values, density=True, bins=50, histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])

ax.legend(loc='best', frameon=False)

plt.show()

val = -0.08168
eps = 0.001
prob_close_to_val = stats.t.cdf(val + eps,**par) - stats.t.cdf(val - eps, **par)
print(f"probability of being close to {val}: {prob_close_to_val * 100:.2f} %")
# probability of being close to -0.08168: 0.41 %

####################################################33


