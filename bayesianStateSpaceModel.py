
import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
from matplotlib import pyplot as plt
import theano
import theano.tensor as tt
import datetime as dt
import matplotlib
import matplotlib.dates as mdates
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return dt.date(year, month, day)

def findMAP(model):
    map_estimate = pm.find_MAP(model=model, method="powell")
    return map_estimate


def showRhat(trace):
    rhat = az.rhat(trace)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax = (rhat.max().to_array().to_series().plot(kind="barh"))
    ax.axvline(1, c="k", ls="--");
    ax.set_xlabel(r"$\hat{R}$");
    ax.invert_yaxis();
    ax.set_ylabel(None);
    plt.show()


def showSummary(summary,index2show='sigma_beta'):
    mask2 = [index2show in x for x in summary.index]
    return summary[mask2]


def gradientPlot(ppc,y_true,coords,like_varname='likelihood',yhat_varname='yhat_fut'):
    percs = np.linspace(51, 99, 100)
    cmap = plt.get_cmap("plasma")
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    mosaic = """AAAA
                BBBB"""
    fig, axs = plt.subplot_mosaic(mosaic, sharex=False, figsize=(20, 10))
    axs = [axs[k] for k in axs.keys()]
    t_data_fut = coords["obs_id_fut"]
    t_data = coords["obs_id"]
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(ppc[like_varname],p,axis=0,)
        lower = np.percentile(ppc[like_varname],100 - p,axis=0,)
        color_val = colors[i]
        axs[0].fill_between(x=t_data,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
        upper = np.percentile(ppc[yhat_varname],p,axis=0,)
        lower = np.percentile(ppc[yhat_varname],100 - p,axis=0,)
        axs[0].fill_between(x=t_data_fut,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
    axs[0].plot(ppc[like_varname].mean(axis=0),color="cyan",label="Mean",)
    axs[0].scatter(x=t_data,y=y_true,color="k",label="Observed Data points",)
    axs[0].plot(t_data_fut,ppc[yhat_varname].mean(axis=0),color="cyan",label="Mean Predicted",)
    axs[0].set_title("Posterior Predictive Fit & Forecast", fontsize=20)
    axs[0].legend()
    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model), ax=axs[1]);plt.show()
    plt.show()



def gradientPlot_compnents(ppc,y_true,coords,varnames=[('likelihood','yhat_fut'),('trend','trend_fut'),('seasonality','seasonality_fut'),('ar1','ar1_fut')],comp=['Fit & Forecast','Trend','Seasonality','Autoregressiv']) :
    percs = np.linspace(51, 99, 100)
    cmap = plt.get_cmap("plasma")
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    mosaic = """AAAA
                BBBB
                CCCC
                DDDD"""
    fig, axs = plt.subplot_mosaic(mosaic, sharex=False, figsize=(20, 10))
    axs = [axs[k] for k in axs.keys()]
    t_data_fut = coords["obs_id_fut"]
    t_data = coords["obs_id"]
    for j,vn in enumerate(varnames):
        if comp[j] == 'Autoregressiv': AR = True
        else: AR = False
        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(ppc[vn[0]],p,axis=0,)
            lower = np.percentile(ppc[vn[0]],100 - p,axis=0,)
            color_val = colors[i]
            axs[j].fill_between(x=t_data,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
            if AR:
                upper = np.percentile(ppc[vn[1]][:,1:],p,axis=0,)
                lower = np.percentile(ppc[vn[1]][:,1:],100 - p,axis=0,)
            else:
                upper = np.percentile(ppc[vn[1]],p,axis=0,)
                lower = np.percentile(ppc[vn[1]],100 - p,axis=0,)
            axs[j].fill_between(x=t_data_fut,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
        axs[j].plot(t_data,ppc[vn[0]].mean(axis=0),color="cyan",label="Mean",)
        if j in [0,1]: axs[j].scatter(x=t_data,y=y_true,color="k",label="Observed Data points",)
        if AR: axs[j].plot(t_data_fut,ppc[vn[1]][:,1:].mean(axis=0),color="cyan",label="Mean Predicted",) 
        else: axs[j].plot(t_data_fut,ppc[vn[1]].mean(axis=0),color="cyan",label="Mean Predicted",)
        axs[j].set_title(comp[j], fontsize=20)
        axs[j].legend()
    plt.show()





save_path = 'your_path'
data = pd.read_csv(save_path + 'data3.csv',delimiter=',') 
data = data.dropna()

y = data["births"].to_numpy()

y = np.log(y)

new_index = [dt.date.fromisoformat(x) for x in data['Date'].values]
data.index = new_index


data["Month"] = new_index
data["Month"] = data["Month"].astype('datetime64')




horizon = 12 # forecast next 12 months
num_obs = len(data)
prediction_length = num_obs + horizon


future_index = new_index
lastdate = future_index[-1]
for i in range(horizon):
    future_index.append(add_months(lastdate, i+1))


future = pd.DataFrame()
future.index = future_index
future['y'] = np.nan
future['y'].iloc[:num_obs] = data["births"].to_numpy()


cycle = 12 # months 

t_data = list(range(num_obs))
n_order = 10
periods = np.array(t_data) / cycle

fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)


######## https://github.com/juanitorduz/btsa/blob/master/python/intro_forecasting/unobserved_components.ipynb  

import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

from statsmodels.tsa.statespace.structural import UnobservedComponents

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100


n_train = len(data)
n_test = 12

data_train_df = future[: n_train]
data_test_df = future[- n_test :]

y_train = data_train_df['y']
y_test = data_test_df['y']









model_params =  {
    'endog': y_train,
    #'endog': y,
    #'exog': x_train,
    'level': 'local level',
    'freq_seasonal': [
         {'period': 12, 'harmonics': 4}
    ],
    'autoregressive': 1,
    'mle_regression': False,
 }
 
model = UnobservedComponents(**model_params)
result = model.fit(disp=0)
result.summary()




"""
                                Unobserved Components Results                                
=============================================================================================
Dep. Variable:                                births   No. Observations:                   48
Model:                                   local level   Log Likelihood                -291.777
                   + stochastic freq_seasonal(12(4))   AIC                            593.554
                                             + AR(1)   BIC                            601.872
Date:                               Sun, 19 Mar 2023   HQIC                           596.539
Time:                                       01:43:47                                         
Sample:                                   01-01-2019                                         
                                        - 12-01-2022                                         
Covariance Type:                                 opg                                         
==============================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------
sigma2.irregular               0.0012   4.28e+04   2.76e-08      1.000   -8.39e+04    8.39e+04
sigma2.level                1.152e+04   5103.800      2.257      0.024    1514.272    2.15e+04
sigma2.freq_seasonal_12(4)  1.496e-07    410.689   3.64e-10      1.000    -804.935     804.935
sigma2.ar                   4.427e+04   3.81e+04      1.163      0.245   -3.03e+04    1.19e+05
ar.L1                         -0.6248      0.273     -2.292      0.022      -1.159      -0.091
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 5.09
Prob(Q):                              0.99   Prob(JB):                         0.08
Heteroskedasticity (H):               1.29   Skew:                             0.51
Prob(H) (two-sided):                  0.66   Kurtosis:                         4.45
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""







"""  for log values !!!!!
                                Unobserved Components Results                                
=============================================================================================
Dep. Variable:                                     y   No. Observations:                   48
Model:                                   local level   Log Likelihood                  60.143
                   + stochastic freq_seasonal(12(4))   AIC                           -110.286
                                             + AR(1)   BIC                           -101.968
Date:                               Sun, 19 Mar 2023   HQIC                          -107.301
Time:                                       01:13:20                                         
Sample:                                            0                                         
                                                - 48                                         
Covariance Type:                                 opg                                         
==============================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------
sigma2.irregular               0.0011      0.001      1.940      0.052   -1.09e-05       0.002
sigma2.level                   0.0002      0.000      0.903      0.366      -0.000       0.000
sigma2.freq_seasonal_12(4)  2.576e-11    8.9e-06   2.89e-06      1.000   -1.74e-05    1.74e-05
sigma2.ar                   1.396e-11      0.001   2.24e-08      1.000      -0.001       0.001
ar.L1                          0.5993   2.17e-14   2.77e+13      0.000       0.599       0.599
===================================================================================
Ljung-Box (L1) (Q):                   3.42   Jarque-Bera (JB):                 0.39
Prob(Q):                              0.06   Prob(JB):                         0.82
Heteroskedasticity (H):               1.16   Skew:                            -0.05
Prob(H) (two-sided):                  0.80   Kurtosis:                         3.48
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
[2] Covariance matrix is singular or near-singular, with condition number 1.35e+38. Standard errors may be unstable.
"""


result.plot_diagnostics(figsize=(12, 9));plt.show()

result.plot_components(alpha=0.05,legend_loc='upper left',figsize=(16, 12));plt.show()

print(f'''
k_states = {model.k_states}
unused_state = {int(model._unused_state)}
ar_order = {model.ar_order}
burnout_obs = {
    model.k_states - int(model._unused_state) - model.ar_order
}
''')

k_states = 10
unused_state = 0
ar_order = 1
burnout_obs = 9


print(f'''
level = {int(model.level)}
k_freq_seas_states = {model._k_freq_seas_states}
ar_order = {model.ar_order}
k_exog = {int((not model.mle_regression) * model.k_exog)}
k_states = {
    model.level + model._k_freq_seas_states + model.ar_order + (not model.mle_regression) * model.k_exog
}
''')

level = 1
k_freq_seas_states = 8
ar_order = 1
k_exog = 0
k_states = 10




##############  Predictions
# We can now generate predictions on the training and test sets (with prediction intervals):





predictions_df = result.get_prediction(steps=n_train).summary_frame(alpha=0.95)

forecast_df = result.get_forecast(steps=n_test).summary_frame(alpha=0.95)

predictions_df.iloc[0] = predictions_df.iloc[1] # bfill 

repetitions = 100

simulations_train_df = result.simulate(anchor='start',nsimulations=n_train,repetitions=repetitions)

simulations_test_df = result.simulate(anchor='end',nsimulations=n_test,repetitions=repetitions)

# Verify expected shape of the simulations dataframes.
assert simulations_train_df.shape == (n_train, repetitions)
assert simulations_test_df.shape == (n_test, repetitions)

#We can compute some statistics from the simulated samples:

y_train_pred_mean = simulations_train_df.mean(axis=1)
y_train_pred_std = simulations_train_df.std(axis=1)
y_train_pred_plus = y_train_pred_mean + 2 * y_train_pred_std
y_train_pred_minus = y_train_pred_mean - 2 * y_train_pred_std

y_test_pred_mean = simulations_test_df.mean(axis=1)
y_test_pred_std = simulations_test_df.std(axis=1)
y_test_pred_plus = y_test_pred_mean + 2 * y_test_pred_std
y_test_pred_minus = y_test_pred_mean - 2 * y_test_pred_std


fig, ax = plt.subplots()
sns.lineplot(x=y_train.index, y=y_train, marker='o', markersize=5, color=sns_c[0], label='y_train', ax=ax)
sns.lineplot(x=y_test.index, y=y_test, marker='o', markersize=5, color=sns_c[1], label='y_test', ax=ax)
ax.axvline(x=y_train.tail(1).index[0], color=sns_c[6], linestyle='--', label='train-test-split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Train-Test Split');plt.show()






fig, ax = plt.subplots()
# Input data
sns.lineplot(x=y_train.index,y=y_train,marker='o',markersize=5,color=sns_c[0],label='births',ax=ax)
sns.lineplot(x=y_test.index,y=y_test,marker='o',markersize=5,color=sns_c[1],label='y_test',ax=ax)
ax.axvline(x=y_train.tail(1).index[0],color=sns_c[6],linestyle='--',label='train-test-split')
# Simulations
for col in simulations_test_df.columns:
    sns.lineplot(
        x=simulations_test_df.index,
        y=simulations_test_df[col],
        color=sns_c[3],
        alpha=0.05, 
        ax=ax
    )

# Prediction intervals
ax.fill_between(x=y_train.index[10: ],y1=predictions_df['mean_ci_lower'][10: ],y2=predictions_df['mean_ci_upper'][10: ],color=sns_c[2],alpha=0.8)
ax.fill_between(x=y_test.index,y1=forecast_df['mean_ci_lower'],y2=forecast_df['mean_ci_upper'],color=sns_c[3],alpha=0.8)

# Predictions
sns.lineplot(x=y_train.index,y=predictions_df['mean'],marker='o',markersize=5,color=sns_c[2],label='y_train_pred',ax=ax)
sns.lineplot(x=y_test.index,y=forecast_df['mean'],marker='o',markersize=5,color=sns_c[3],label='y_test_pred',ax=ax)
ax.axvline(x=y_train.index[10],color=sns_c[5],linestyle='--',label='diffuse initialization')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Births predicted for 2023 by Unobserved Components Model Predictions')
#plt.savefig(f'images/unobserved_components_predictions.png', dpi=200, bbox_inches='tight');
plt.show()





forecasts = forecast_df.copy()

for col in forecasts.columns:
    forecasts[col] = forecasts[col].astype(int)

forecasts

births      mean  mean_se  mean_ci_lower  mean_ci_upper
2023-01-01  8173      325           8152           8193
2023-02-01  8453      325           8433           8474
2023-03-01  8525      369           8502           8548
2023-04-01  9139      376           9116           9163
2023-05-01  9222      398           9197           9247
2023-06-01  9204      407           9178           9229
2023-07-01  9401      424           9375           9428
2023-08-01  9057      433           9030           9084
2023-09-01  8586      446           8558           8614
2023-10-01  8247      456           8218           8275
2023-11-01  7519      464           7490           7548
2023-12-01  7448      477           7418           7478




###################################################################################### bayesian model ############

y_true = data["births"].to_numpy()

y_log = np.log(y_true)






coords= {
    "coefs": {"mu": np.array([0.6, 0.1]), "sigma": np.array([0.5, 0.03]), "size": 2},
    "alpha": {"mu": 8.77, "sigma": 0.1}, # y.mean()  9.13452597789831
    "beta": {"mu": 0., "sigma": 0.01},
    "sigma": 8.,
    "beta_fourier": {"mu": 0, "sigma": 2},
    "init": {"mu": 0., "sigma": 0.05, "size": 1},
    "obs_id": t_data,
    "num_obs":num_obs,
    "obs_id_fut_1": np.arange(num_obs - 1,prediction_length),
    "obs_id_fut": np.arange(num_obs,prediction_length),
    "fourier_features": np.arange(2 * n_order),
    }






n_order = 10
periods_fut = (num_obs + np.arange(horizon)) / cycle

fourier_features_new = pd.DataFrame(
    {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods_fut * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )






model = pm.Model(coords=coords)
with model:
    t = pm.Data("t", t_data)
    y = pm.Data("y",y_log)
    ar_factor = pm.Normal("ar_factor", mu=0.1, sigma=0.1,shape=1)
    coefs = pm.Normal("coefs", mu=coords["coefs"]["mu"], sigma=np.array(coords["coefs"]["sigma"]),shape=2)
    sigma = pm.HalfNormal("sigma", coords["sigma"])
    init = pm.Normal.dist(coords["init"]["mu"], coords["init"]["sigma"], shape=coords["init"]["size"])
    ar1 = pm.AR("ar1",coefs,sigma=sigma,init=init,constant=True, shape=coords["num_obs"])
    alpha = pm.Normal("alpha", coords["alpha"]["mu"], coords["alpha"]["sigma"])
    beta_sigma = pm.HalfNormal("beta_sigma", sigma=coords["beta"]["sigma"])
    beta_mu = pm.Normal("beta_mu", mu=coords["beta"]["mu"],sigma=coords["beta"]["sigma"])
    beta = pm.GaussianRandomWalk("beta", mu=beta_mu, sigma=beta_sigma,shape=coords["num_obs"])
    trend = pm.Deterministic("trend", alpha + beta * t, dims="obs_id")
    beta_fourier = pm.Normal("beta_fourier",mu=coords["beta_fourier"]["mu"],sigma=coords["beta_fourier"]["sigma"], dims="fourier_features")
    seasonality = pm.Deterministic("seasonality", pm.math.dot(beta_fourier, fourier_features.to_numpy().T)) # , dims="fourier_features",
    mu = ar1*ar_factor + trend + seasonality
    #mu = ar1 + trend * (1 + seasonality)
    outcome = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_log, shape=coords["num_obs"]) # dims="obs_id"
    t_fut = pm.Data("t_fut", coords["obs_id_fut"] )
    components = pm.Normal.dist(mu=ar1[..., -1], sigma=1, shape=coords["num_obs"])
    ar1_fut = pm.AR("ar1_fut",init=components,rho=coefs,sigma=sigma,constant=True,dims="obs_id_fut_1",)


start = findMAP(model)



with model:
    trace = pm.sample(4000, random_seed=100, target_accept=0.95,return_inferencedata=True)


with model:
    beta_init = pm.Normal.dist(mu=beta[..., -1], sigma=1, shape=coords["num_obs"])
    #beta_fut = pm.Normal("beta_fut", mu=coords["beta"]["mu"], sigma=coords["beta"]["sigma"], init=beta_init,dims="obs_id_fut")
    beta_fut = pm.GaussianRandomWalk("beta_fut", mu=0., sigma=beta_sigma, init=beta_init,dims="obs_id_fut")
    seasonality_fut = pm.Deterministic("seasonality_fut", pm.math.dot(beta_fourier, fourier_features_new.to_numpy().T), dims="obs_id_fut")
    trend_fut = pm.Deterministic("trend_fut", alpha +  (beta[-1] + beta_fut) * t_fut, dims="obs_id_fut")
    #mu_fut = ar1_fut[1:] + trend_fut * (1 + seasonality_fut)
    mu_fut = ar1_fut[1:]*ar_factor + trend_fut  + seasonality_fut
    yhat_fut = pm.Normal("yhat_fut", mu=mu_fut, sigma=sigma, dims="obs_id_fut")





az.plot_trace(trace, figsize=(16, 10));plt.show()
summary = az.summary(trace)
showSummary(summary,index2show='beta_sigma')
showRhat(trace)

ppc = pm.sample_posterior_predictive(trace, model=model,var_names=["likelihood","yhat_fut","trend","trend_fut","seasonality","seasonality_fut","ar1","ar1_fut"])


gradientPlot_compnents(ppc=ppc,y_true=y_log,coords=coords,varnames=[('likelihood','yhat_fut'),('trend','trend_fut'),('seasonality','seasonality_fut'),('ar1','ar1_fut')],comp=['Fit & Forecast','Trend','Seasonality','Autoregressiv'])


ppc_real = ppc.copy()

for v in ['likelihood','yhat_fut','trend','trend_fut','seasonality','seasonality_fut']:
    ppc_real[v] = np.exp(ppc_real[v])


time_dict= {
    "obs_id_fut": future.index[num_obs:],
    "obs_id": future.index[:num_obs],
    }



gradientPlot_compnents(ppc=ppc_real,y_true=y_true,coords=time_dict,varnames=[('likelihood','yhat_fut'),('trend','trend_fut'),('seasonality','seasonality_fut'),('ar1','ar1_fut')],comp=['Fit & Forecast','Trend','Seasonality','Autoregressiv'])




def gradientPlot_compnents2(ppc,y_true,coords,varnames=[('likelihood','yhat_fut'),('trend','trend_fut'),('seasonality','seasonality_fut')],comp=['Fit & Forecast','Trend','Seasonality']) :
    percs = np.linspace(51, 99, 100)
    cmap = plt.get_cmap("viridis") # "plasma", "viridis" "magma"
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    if len(varnames) == 3: 
        mosaic = """AAAA
                    BBBB
                    CCCC"""
    if len(varnames) == 1: 
        mosaic = """AAAA"""
    fig, axs = plt.subplot_mosaic(mosaic, sharex=False, figsize=(20, 10))
    axs = [axs[k] for k in axs.keys()]
    t_data_fut = coords["obs_id_fut"]
    t_data = coords["obs_id"]
    out_dict = {}
    for j,vn in enumerate(varnames):
        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(ppc[vn[0]],p,axis=0,)
            lower = np.percentile(ppc[vn[0]],100 - p,axis=0,)
            color_val = colors[i]
            axs[j].fill_between(x=t_data,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
            upper = np.percentile(ppc[vn[1]],p,axis=0,)
            lower = np.percentile(ppc[vn[1]],100 - p,axis=0,)
            axs[j].fill_between(x=t_data_fut,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
            if vn[1] == 'yhat_fut':
                out_dict[str(p)] = upper.flatten()
                out_dict[str(100-p)] = lower.flatten()
        axs[j].plot(t_data,ppc[vn[0]].mean(axis=0),color="cyan",label="Mean",)
        if j in [0,1]: axs[j].scatter(x=t_data,y=y_true,color="k",label="Observed Data points",)
        axs[j].plot(t_data_fut,ppc[vn[1]].mean(axis=0),color="cyan",label="Mean Predicted",)
        if round(p) == 95:
            axs[j].plot(t_data_fut,upper.flatten(),color="red",label="95%",)
            axs[j].plot(t_data_fut,lower.flatten(),color="red",label="5%",)
            out_dict['ci_upper'] = upper.flatten()
            out_dict['ci_lower'] = lower.flatten()
        if vn[1] == 'yhat_fut':
            out_dict['mean'] = ppc[vn[1]].mean(axis=0)
        axs[j].set_title(comp[j], fontsize=20)
        axs[j].legend()
    plt.show()
    return out_dict


out_dict = gradientPlot_compnents2(ppc=ppc_real,y_true=y_true,coords=time_dict,varnames=[('likelihood','yhat_fut'),('trend','trend_fut'),('seasonality','seasonality_fut')],comp=['Fit & Forecast','Trend','Seasonality'])

out_dict = gradientPlot_compnents2(ppc=ppc_real,y_true=y_true,coords=time_dict,varnames=[('likelihood','yhat_fut')],comp=['Fit & Forecast'])




def gradientPlot_compnents3(ppc,y_true,coords,varnames=[('likelihood','yhat_fut')],comp=['Fit 2019-2022 & Forecast for births per month 2023']) :
    percs = np.linspace(51, 99, 100)
    cmap = plt.get_cmap("viridis") # "plasma", "viridis" "magma"
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    if len(varnames) == 3: 
        mosaic = """AAAA
                    BBBB
                    CCCC"""
    if len(varnames) == 1: 
        mosaic = """AAAA"""
    fig, axs = plt.subplot_mosaic(mosaic, sharex=False, figsize=(20, 10))
    axs = [axs[k] for k in axs.keys()]
    t_data_fut = coords["obs_id_fut"]
    t_data = coords["obs_id"]
    out_dict = {}
    for j,vn in enumerate(varnames):
        for i, p in enumerate(percs[::-1]):
            color_val = colors[i]
            upper = np.percentile(ppc[vn[1]],p,axis=0,)
            lower = np.percentile(ppc[vn[1]],100 - p,axis=0,)
            axs[j].fill_between(x=t_data_fut,y1=upper.flatten(),y2=lower.flatten(),color=cmap(color_val),alpha=0.1,)
            out_dict[str(p)] = upper.flatten()
            out_dict[str(100-p)] = lower.flatten()
            if round(p,1) == 95.1:
                axs[j].plot(t_data_fut,upper.flatten(),color="red",label="95%",)
                axs[j].plot(t_data_fut,lower.flatten(),color="red",label="5%",)
                out_dict['ci_upper'] = upper.flatten()
                out_dict['ci_lower'] = lower.flatten()
        axs[j].plot(t_data_fut,ppc[vn[1]].mean(axis=0),color="cyan",label="Mean Predicted",)
        out_dict['mean'] = ppc[vn[1]].mean(axis=0)
        axs[j].set_title(comp[j], fontsize=20)
        axs[j].legend()
    plt.show()
    return out_dict


out_dict = gradientPlot_compnents3(ppc=ppc_real,y_true=y_true,coords=time_dict,varnames=[('likelihood','yhat_fut')],comp=['Fit 2019-2022 & Forecast for births per month 2023'])






bayes_future = pd.DataFrame(out_dict)

bayes_future[['mean','ci_lower','ci_upper']]









