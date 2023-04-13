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
|2022-12   |  6748  | 7017 |  7305 |  7522 |
|2023-01   |  7849  | 8159 |  8437 |  8277 |
|2023-02   |  7281  | 7590 |  7889 |  7921 |
|2023-03   |  8085  | 8414 |  8752 |  not released|
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
|2022-12   |  6836  |  7277|   7719|   7522 |
|2023-01   |  7469  |  7963|   8456|   8277 |
|2023-02   |  7113  |  7607|   8102|  7921 |
|2023-03   |  7728  |  8223|   8718|  not released|
|2023-04   |  7363  |  7857|   8352|  |
|2023-05   |  8087  |  8581|   9076|  |
|2023-06   |  8018  |  8512|   9007|  |
|2023-07   |  7793  |  8288|   8783|  |
|2023-08   |  7911  |  8406|   8901|  |
|2023-09   |  7002  |  7497|   7992|  |
|2023-10   |  6804  |  7299|   7793|  |
|2023-11   |  6167  |  6662|   7157|  |


# Poisson Regression

To investigate causes of dropping birth count different regressors are used in a Poisson Regression Model.

100 random selecteded samples were used for training and testing. (see poisson2.py)

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

Migration had no impact on births during this period (jan/2019 - nov/2022)
- cpi = -0.01 (pvalue:0.01) means: higher inflation has (little) negative effekt on births
- jabs = -0.05 (pvalue:0.02): more jabs -> less births. Model is quite confident about that. 

![jabs](https://github.com/marl2en/predictFuture/blob/main/simulated_vaccination_status_pregnant.png)



# Rolling Regression
Instead of a linear reggression with statsmodel we choose PYMC to analyse possible causes for a dropping birth rate.
"PyMC is a Python package for Bayesian statistical modeling and probabilistic machine learning." (Wikipedia)

Instead of the number of births per month we use: y = (number of births per month)/(number of women in fertile age).
Women in fertile age: 15-49 is about 2.2 millions and rising. (data3.csv: Women)
Because the numbers of births is now diveded by the number of women in fertile age, we don't use population size or migration surplus as a variable. 
The mean age of the population didn't changed that much in four years (2019-2022). 

Variables used:

--- known Unknowns (factors suspected to have an influence, but uncertain how much) ---
- CPI (Consumer Price Index) - economic hardship for young families (beta_cpi)
- Vaccination status (beta_jabs)

--- unknown Unknowns ---
- beta_unknown

These variables are changing with time. Modeled as Gaussian Randomwalks.
![model](https://github.com/marl2en/predictFuture/blob/main/model2.png)

![fittedmodel](https://github.com/marl2en/predictFuture/blob/main/fittedModel.png)
The model fits well. Trend and seasonality look fine. 

![tracePlot](https://github.com/marl2en/predictFuture/blob/main/trace_plot.png)
Plot of model parameters trace. 

For more information, see rhat and summary3.csv.

![tracePlot](https://github.com/marl2en/predictFuture/blob/main/beta_cpi.png)


Inflation (CPI) had a little negative effect on births since 2021. 

![beta_jabs](https://github.com/marl2en/predictFuture/blob/main/beta_jabs.png)

Estimated effect of mRNA vaccins on births. 

![beta_jabs](https://github.com/marl2en/predictFuture/blob/main/beta_unknown.png)
Effect of unknowns. 


# State Space Model (SSM)
Monthly births 2019-2022. Predict births for 2023 with:
- SSM with UnobservedComponents of statsmodel library. 
- structural timeseries (trend,seasonality,autoregressive) with pymc3 (bayesianStateSpaceModel.py)

![tracePlot](https://github.com/marl2en/predictFuture/blob/main/ssm_bayesian_forecast2023viridisCI.png)

Gradient plot

|Year-Month|  lower (5%) | mean | upper (95%) |  real |
|----------|--------|------|-------|-------|
|2023-01   |  8114 |  8532|   8955|  8277 |
|2023-02   |  7605  |  8056|   8539|  7921 |
|2023-03   |  8247  |  8833|   9473|  |
|2023-04   |  8118  |  8779|   9517|  |
|2023-05   |  8465  |  9263|  10133|  |
|2023-06   |  8177  |  9034|   9982|  |
|2023-07   |  8239  |  9177|  10241|  |
|2023-08   |  8063  |  9065|  10204|  |
|2023-09   |  7357  |  8343|   9464|  |
|2023-10   |  7194  |  8241|   9428|  |
|2023-11   |  6461  |  7463|   8623|  |
|2023-01   |  6237  |  7266|   8438|  |





!!! For more information look at the pdf file !!!
