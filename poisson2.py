
"""
sudo pip3 install convertdate 
sudo pip3 install lunarcalendar
sudo pip3 install holidays

#sudo pip3 install fbprophet
sudo pip3 install prophet==1.0

sudo pip3 install Theano

sudo apt-get install hdf5-devel

"""

# Association of SARS-CoV-2 Vaccination During Pregnancy With Pregnancy Outcomes
# https://jamanetwork.com/journals/jama/fullarticle/2790608




import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
#import json

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


def showDTplot(df,col2show=['realbirths','yhat','yhat_lower', 'yhat_upper'],offset=2,title='Live Births in Sweden',ylabel='Births per Month',confInt=[],target='Live births',horizon=12,year=False):
    colors = ['b','g','r','yellow','black']
    fig, axes = plt.subplots(1,1,sharex=True,figsize=(16,10))
    fig.autofmt_xdate()
    axes.set_title(title,fontsize=14)
    axes.set_ylabel(ylabel, fontsize=12)
    if year: axes.set_xlabel("Year", fontsize=12)
    else: axes.set_xlabel("Time in Month-Year", fontsize=12)
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



months = ['January','February','March','April','May','June','July','August','September','October','November','December']



# your path to documents
save_path = '/home/pi/Documents/BirthRate/'

# forecast2.to_csv('/home/usix/Documents/BirthRate/births_month_forecast.csv',index=False,header=True,sep=',')


births_month = pd.read_csv(save_path + 'births_month3.csv',delimiter=',') 

births_month.head()
births_month.tail()

births_month.columns
Index(['Year', 'Month', 'Population at the end of period',
       'Population growth1', 'Live births', 'Still births', 'Deaths',
       'Population surplus', 'Immigrants', 'Emigrants',
       'Surplus of immigrants', 'Date'],
      dtype='object')

cpi = pd.read_csv(save_path + 'cpi/CPI_annualChanges.csv',delimiter=',')  
startdate = dt.date.fromisoformat(births_month['Date'].iloc[0])
enddate = dt.date.fromisoformat(births_month['Date'].iloc[-1])

cpi_list = []
for col in cpi.columns:
    y,m = col.split('M')
    act_date = dt.date(int(y), int(m), 1)
    if startdate <= act_date <= enddate:
        cpi_list.append([act_date.isoformat(),cpi[col].values[0]])



## select data
data = births_month[['Year','Month','Live births','Population at the end of period','Surplus of immigrants']].copy()

# Rename columns
data.columns = ['year','month','births','population','surplus']

new_index = [dt.date.fromisoformat(x) for x in births_month['Date'].values]

data.index = new_index

# population in millions 
data['population'] = data['population'].values * 1.e-6

# convert months tu number
mon = [months.index(m)+1.0 for m in data['month'].values] # jan=1, convert to float

data['month'] = mon

data['year'] = data['year'].values - 2000. # years after 2000 in float

data['Date'] = births_month['Date'].values
data['cpi'] = [x[1] for x in cpi_list]

data.to_csv(save_path + 'data.csv',index=False,header=True,sep=',')



### simulate jab data 



def simulateJab(scale=6.0,bins=10,endval=0.8,show=True):
    # endvalue: value at the end of the period
    #jabs = np.random.randn(loc=0.0, scale=6.0, size=500)
    jabs=np.random.normal(loc=0.0, scale=scale, size=1000)
    count, bins_count = np.histogram(jabs, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)*endval
    if show:
        plt.plot(bins_count[1:], pdf, color="r", label="PDF")
        plt.plot(bins_count[1:], cdf, color="g",label="CDF")
        plt.legend()
        plt.show()
    return cdf

num_per = 10 

jab1 = simulateJab(scale=6.0,bins=num_per,endval=0.8,show=True)# more than 80% 1-jabbed and end of period

jab2 = simulateJab(scale=4.0,bins=num_per,endval=0.7,show=True)# more than 70% 2-jabbed and end of period

jab3 = simulateJab(scale=2.0,bins=num_per,endval=0.45,show=True)# around 45% 3-jabbed and end of period

mask = (dt.date(2021,4,1) <= data.index) & (data.index <= dt.date(2021,5,1))

ix = np.argmax(mask)

jabs_startdate = [dt.date(2021,4,1),dt.date(2021,5,1),dt.date(2021,12,1)]
jabs_enddate = [add_months(x, num_per) for x in jabs_startdate]
jabs = [jab1,jab2,jab3]


#data.drop('jab0', inplace=True, axis=1)

i=1
for j,s,e in zip(jabs,jabs_startdate,jabs_enddate):
    data['jab'+str(i)] = 0.
    mask = (s <= data.index) & (data.index < e)
    data['jab'+str(i)][mask] = j
    mask = (data.index >= e)
    data['jab'+str(i)][mask] = [j[-1]]*mask.sum() # constant value 
    i += 1


showDTplot(data,col2show=['jab1','jab2', 'jab3'],offset=24,title='Vaccination Status of pregnant Women in Sweden (simulated',ylabel='in procent, 0,1 = 10%',confInt=[],target='',horizon=12,year=False)

# add all jabs
data['jabs'] = 0.
for j in ['jab1','jab2', 'jab3']:
    data['jabs'] = data['jabs'].values + data[j].values




from sklearn.metrics import mean_squared_error
from patsy import dmatrices
from sklearn.metrics import mean_absolute_percentage_error as mape

# ['year', 'month', 'births', 'population', 'surplus', 'jab1', 'Date','jab2', 'jab3', 'cpi', 'jabs']

expr = """births ~ year  + month + population + surplus + cpi + jabs"""

trainsize = 0.8

mape_list = [] 
params_list = []
pvalue_list = [] 


for i in range(100):
    mask = np.random.rand(len(data)) < trainsize 
    df_train = data[mask]
    df_test = data[~mask]
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    res = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    #res = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=0.1)).fit(method='bfgs', maxiter=5000) # 'bfgs', method='powell'
    predictions = res.get_prediction(X_test)
    psf = predictions.summary_frame()
    mape_list.append(mape(psf['mean'].values,y_test))
    params_list.append(res.params)
    pvalue_list.append(res.pvalues)


error = np.array(mape_list).mean() # 0.052110179207352225

params = {}
for row in res.params.index: params[row] = []

for p in params_list:
    for row in p.index: params[row] += [round(p[row],2)]

for k,v in params.items():
    p = np.array(v)
    print('param:',k,'mean:',round(p.mean(),2),'std:',round(p.std(),2))


param: Intercept mean: 15.93 std: 7.81
param: year mean: 0.09 std: 0.06
param: month mean: 0.0 std: 0.01
param: population mean: -0.83 std: 0.87
param: surplus mean: 0.0 std: 0.0
param: cpi mean: -0.01 std: 0.0
param: jabs mean: -0.05 std: 0.02



pvalues = {}
for row in res.pvalues.index: pvalues[row] = []

for p in pvalue_list:
    for row in p.index: pvalues[row] += [round(p[row],2)]

for k,v in pvalues.items():
    p = np.array(v)
    print('param:',k,'mean:',round(p.mean(),2),'std:',round(p.std(),2))

param: Intercept mean: 0.04 std: 0.15
param: year mean: 0.07 std: 0.17
param: month mean: 0.27 std: 0.33
param: population mean: 0.15 std: 0.25
param: surplus mean: 0.0 std: 0.0
param: cpi mean: 0.01 std: 0.06
param: jabs mean: 0.02 std: 0.11



res.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                 births   No. Observations:                   39
Model:                            GLM   Df Residuals:                       32
Model Family:                 Poisson   Df Model:                            6
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -756.94
Date:                Mon, 13 Feb 2023   Deviance:                       1086.1
Time:                        14:59:37   Pearson chi2:                 1.08e+03
No. Iterations:                     4   Pseudo R-squ. (CS):              1.000
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     18.7726      2.496      7.522      0.000      13.881      23.664
year           0.1076      0.019      5.553      0.000       0.070       0.146
month          0.0023      0.002      1.238      0.216      -0.001       0.006
population    -1.1435      0.279     -4.101      0.000      -1.690      -0.597
surplus     2.004e-05   8.46e-07     23.703      0.000    1.84e-05    2.17e-05
cpi           -0.0093      0.001     -7.528      0.000      -0.012      -0.007
jabs          -0.0439      0.007     -6.353      0.000      -0.057      -0.030
==============================================================================
"""
>>> res.summary2()
<class 'statsmodels.iolib.summary2.Summary'>
"""
              Results: Generalized linear model
==============================================================
Model:              GLM              AIC:            1527.8781
Link Function:      Log              BIC:            968.8391 
Dependent Variable: births           Log-Likelihood: -756.94  
Date:               2023-02-13 15:01 LL-Null:        -1493.5  
No. Observations:   39               Deviance:       1086.1   
Df Model:           6                Pearson chi2:   1.08e+03 
Df Residuals:       32               Scale:          1.0000   
Method:             IRLS                                      
--------------------------------------------------------------
                Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
--------------------------------------------------------------
Intercept      18.7726   2.4956  7.5224 0.0000 13.8814 23.6638
year            0.1076   0.0194  5.5531 0.0000  0.0696  0.1456
month           0.0023   0.0018  1.2383 0.2156 -0.0013  0.0059
population     -1.1435   0.2788 -4.1007 0.0000 -1.6900 -0.5970
surplus         0.0000   0.0000 23.7030 0.0000  0.0000  0.0000
cpi            -0.0093   0.0012 -7.5278 0.0000 -0.0117 -0.0069
jabs           -0.0439   0.0069 -6.3534 0.0000 -0.0575 -0.0304
==============================================================
"""

res = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
res.summary2()
<class 'statsmodels.iolib.summary2.Summary'>
"""
               Results: Generalized linear model
===============================================================
Model:               GLM              AIC:            927.7382 
Link Function:       Log              BIC:            -117.2223
Dependent Variable:  births           Log-Likelihood: -456.87  
Date:                2023-02-13 15:12 LL-Null:        -456.88  
No. Observations:    39               Deviance:       0.011702 
Df Model:            6                Pearson chi2:   0.0117   
Df Residuals:        32               Scale:          1.0000   
Method:              IRLS                                      
---------------------------------------------------------------
            Coef.  Std.Err.    z    P>|z|    [0.025     0.975] 
---------------------------------------------------------------
Intercept  18.7790 768.1542  0.0244 0.9805 -1486.7755 1524.3336
year        0.1069   5.9659  0.0179 0.9857   -11.5860   11.7998
month       0.0018   0.5640  0.0033 0.9974    -1.1035    1.1072
population -1.1426  85.8313 -0.0133 0.9894  -169.3688  167.0836
surplus     0.0000   0.0003  0.0776 0.9382    -0.0005    0.0005
cpi        -0.0098   0.3639 -0.0269 0.9785    -0.7231    0.7035
jabs       -0.0407   2.0855 -0.0195 0.9844    -4.1282    4.0467
===============================================================

"""


#### all data at once

df_train = data.copy()
y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')

res = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
res.summary()

"""
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                 births   No. Observations:                   47
Model:                            GLM   Df Residuals:                       40
Model Family:                 Poisson   Df Model:                            6
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -907.37
Date:                Mon, 13 Feb 2023   Deviance:                       1298.8
Time:                        15:20:47   Pearson chi2:                 1.30e+03
No. Iterations:                     4   Pseudo R-squ. (CS):              1.000
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     14.6684      2.205      6.653      0.000      10.347      18.990
year           0.0779      0.017      4.576      0.000       0.045       0.111
month          0.0012      0.002      0.717      0.473      -0.002       0.005
population    -0.6893      0.246     -2.799      0.005      -1.172      -0.207
surplus     2.056e-05    7.9e-07     26.017      0.000     1.9e-05    2.21e-05
cpi           -0.0086      0.001     -7.416      0.000      -0.011      -0.006
jabs          -0.0490      0.006     -8.210      0.000      -0.061      -0.037
==============================================================================
"""



plt.plot(res.fittedvalues)
plt.plot(data.births)
plt.show()
