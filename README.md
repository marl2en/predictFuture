# predictFuture
demonstration of uni-variate time series prediction by predicting monthly births in Sweden for the next 12 months 

# real data
![realdata](https://github.com/marl2en/predictFuture/blob/main/Live_births2019_2022.png)

# change of trend
![change](https://github.com/marl2en/predictFuture/blob/main/Change_of_trend.png)


# prediction for 2023
![IMG_6013](https://github.com/marl2en/predictFuture/blob/main/sarimaxPrediction_with_expSmoothing_trend.png)

# yearly change of births last 100 years
![last100](https://github.com/marl2en/predictFuture/blob/main/birthsPerYear1923_2022proc.png)

# forecast of fbprophet after bug fix
![prophecy](https://github.com/marl2en/predictFuture/blob/main/Prophecy_2023_without_bugs.png)

# fbprophet forecast 12 months
|Year-Month|  lower | mean | upper |  real |
|----------|--------|------|-------|-------|
|2022-12   |  6748  | 7017 |  7305 |  not released |
|2023-01   |  7849  | 8159 |  8437 |  |
|2023-02   |  7281  | 7590 |  7889 |  |
|2023-03   |  8085  | 8414 |  8752 |  |
|2023-04   |  7924  | 8258 |  8592 |  |
|2023-05   |  8394  | 8755 |  9075 |  |
|2023-06   |  8095  | 8470 |  8851 |  |
|2023-07   |  8181  | 8555 |  8930 |  |
|2023-08   |  7973  | 8377 |  8784 |  |
|2023-09   |  7130  | 7546 |  7970 |  |
|2023-10   |  6931  | 7369 |  7820 |  |
|2023-11   |  6067  | 6512 |  6992 |  |

# sarimax forecast 12 months
|Year-Month|  lower | mean | upper |  real |
|----------|--------|------|-------|-------|
|2022-12   |  6836  |  7277|   7719|    not released
|2023-01   |  7469  |  7963|   8456|  |
|2023-02   |  7113  |  7607|   8102|  |
|2023-03   |  7728  |  8223|   8718|  |
|2023-04   |  7363  |  7857|   8352|  |
|2023-05   |  8087  |  8581|   9076|  |
|2023-06   |  8018  |  8512|   9007|  |
|2023-07   |  7793  |  8288|   8783|  |
|2023-08   |  7911  |  8406|   8901|  |
|2023-09   |  7002  |  7497|   7992|  |
|2023-10   |  6804  |  7299|   7793|  |
|2023-11   |  6167  |  6662|   7157|  |


# Poisson Regression

To investigate causes of dropping birth count different regressors are used in a Poisson Regression Model
Random samples are used for training and testing. (100 samples, see poisson2.py)
Mean absolute percentage error (mape) is about 5%. 

Parameters used for the model are:
- year
- month
- population size
- surplus of migration
- CPI (Consumer Price Index)
- Vaccination Status of pregnant women (simulation, see https://jamanetwork.com/journals/jama/fullarticle/2790608)

expr = """births ~ year  + month + population + surplus + cpi + jabs"""

|parameter|  mean | std |
|----------|--------|------|
|Intercept   |  15.93  | 7.81|
|year   |  0.09 |  0.06|
|month   |  0.0  | 0.01|
|population   | -0.83 |  0.87|
|surplus   |  0.0  |  0.0|
|cpi   |  -0.01  |  0.0|
|jabs   |  -0.05  |  0.02|


|pvalues|  mean | std |
|----------|--------|------|
|Intercept   |  0.04  | 0.15|
|year   |  0.07 |  0.17|
|month   | 0.27 | 0.33|
|population   | 0.15 |  0.25|
|surplus   |  0.0  |  0.0|
|cpi   |  0.01 |  0.06|
|jabs   |  0.02  |  0.11|

pvalues under 0.05 are migration surplus, cpi and vaccination status (jabs). 

migration had no effekt on births during this period (jan/2019 - nov/2022)
- cpi = -0.01 means: higher inflation has (little) negative effekt on births
- jabs = -0.05: more jabs -> less births. Model is quite confident about that. 



!!! For more information look at the pdf file !!!
