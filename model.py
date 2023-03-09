

import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy import stats
from statsmodels.tsa.api import SARIMAX
import os
import calendar
from matplotlib.colors import SymLogNorm #, Normalize
from matplotlib.ticker import MaxNLocator


import pymc3 as pm
import seaborn as sns
import arviz as az
import theano
import theano.tensor as tt
from pymc3.distributions.timeseries import GaussianRandomWalk
from scipy import optimize


def saveDict(outdict,filename=''):
    with open(filename, 'w') as outfile:
        json.dump(outdict, outfile,sort_keys=True,indent=4) # separators=(',', ': ')

def loadDict(filename=''):
    with open(filename) as f:
        dictobj = json.load(f)
    return dictobj


def showDTplot(df,col2show=['realbirths','yhat','yhat_lower', 'yhat_upper'],offset=2,title='Live Births in Sweden',ylabel='Births per Month',confInt=[],target='Live births',horizon=12,year=False,showBorder=False,plotname=''):
    plt.close()
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
            try: axes.fill_between(df.index[offset:], df[ele[0]].values[offset:], df[ele[1]].values[offset:],color=colors[i], alpha=0.1)
            except: pass
    for ele in col2show:
        if ele == target: axes.plot(df.index[offset:-horizon], df[ele][offset:-horizon],label=ele)
        else: axes.plot(df.index[offset:], df[ele][offset:],label=ele)
    if year: xfmt = mdates.DateFormatter('%Y')
    else: xfmt = mdates.DateFormatter('%m-%y')
    axes.xaxis.set_major_formatter(xfmt)
    plt.legend()
    plt.grid(axis='both', color='0.95')
    fig.tight_layout()
    if plotname == '': plt.show()
    else: plt.savefig(plotname)



def normolize(arr):
    maxval = arr.max()
    minval = arr.min()
    diff = maxval - minval
    if diff != 0.:
        return (arr - minval)/diff
    else:
        return arr/maxval


def standardize(arr):
    std = arr.std()
    mean = arr.mean()
    if std != 0.:
        return (arr - mean)/std
    else:
        return (arr - mean)


def simpleLinReg(y,N=0,show=False):
    num = len(y)
    x = np.arange(num)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print('m',m,'c',c)
    if N == 0: yy = np.linspace(0.,m * num,num) + c
    else: yy = np.linspace(0.,N * m,N) + c
    if show:
        plt.plot(y,label='y')
        plt.plot(yy,label='lin reg')
        plt.legend()
        plt.show()
    return yy


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


def findMAP(model):
    map_estimate = pm.find_MAP(model=model, method="powell")
    return map_estimate

# https://austinrochford.com/posts/apc-pymc.html
def index_of_dispersion(x):
    return x.var() / x.mean()



months = ['January','February','March','April','May','June','July','August','September','October','November','December']


# your path to documents
save_path = '/home/usix/Documents/BirthRate/'




############################ births per million women in fertile age 15-49 ###########3




RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

az.style.use("arviz-darkgrid")



def _sample(array, n_samples):
    """Little utility function, sample n_samples with replacement"""
    idx = np.random.choice(np.arange(len(array)), n_samples, replace=True)
    return array[idx]



save_path = 'YOUR_PATH'
data = pd.read_csv(save_path + 'data3.csv',delimiter=',') 
data = data.dropna()

data ['Month'] = data['Date'].values
t = np.linspace(0,1,len(data))


# Next, for the target variable, we divide by the maximum. We do this, rather than standardising, so that the sign of the observations in unchanged - this will be necessary for the seasonality component to work properly later on.

y = data["births"].to_numpy() / data["Women"].to_numpy()
data["y"] = y

print('dispersion:',y.var()/y.mean())
# dispersion: 32.032442455240584


y_max = np.max(y)
y = y / y_max

print('dispersion:',y.var()/y.mean())
# dispersion: 0.006734975410480795



showDTplot(data,col2show=['Women'],offset=0,title='women in fertile age 15-49',ylabel='in Millions',confInt=[],target='',horizon=12,year=False,showBorder=False,plotname='')

plt.plot(y);plt.show()


new_index = [dt.date.fromisoformat(x) for x in data['Date'].values]
data.index = new_index


data["Month"] = new_index
data["Month"] = data["Month"].astype('datetime64')

n_order = 10
periods = data["Month"].dt.dayofyear / 365.25

fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)
fourier_features



num_obs = len(data)
t = np.linspace(0,1,num_obs)
x_jabs = standardize(data['jabs'].values)
x_cpi = standardize(data['cpi'].values)




SAMPLE_KWARGS = {
    "draws": 500,
    "tune": 6000,
    'cores': 4,
    #'random_seed': [SEED + i for i in range(CHAINS)],
    'return_inferencedata': True,
    'target_accept': 0.90
    }



modelx = theano.shared(t)

sigma=0.005

coords = {"fourier_features": np.arange(2 * n_order)}
with pm.Model(check_bounds=False, coords=coords) as model:
    α = pm.Normal("α", mu=0, sigma=0.5)
    beta_jabs = pm.GaussianRandomWalk("beta_jabs", mu=0, sigma=sigma,shape=num_obs)
    beta_cpi = pm.GaussianRandomWalk("beta_cpi", mu=0, sigma=sigma,shape=num_obs)
    beta_unknown = pm.GaussianRandomWalk("beta_unknown", mu=0, sigma=sigma,shape=num_obs)
    beta = pm.Deterministic("beta",  beta_jabs * x_jabs + beta_cpi * x_cpi + beta_unknown)
    trend = pm.Deterministic("trend", α + beta * modelx)
    β_fourier = pm.Normal("β_fourier", mu=0, sigma=0.1, dims="fourier_features")
    seasonality = pm.Deterministic("seasonality", pm.math.dot(β_fourier, fourier_features.to_numpy().T))
    μ = trend * (1 + seasonality)
    σ = pm.HalfNormal("σ", sigma=0.1)
    pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)
    prior_predictive = pm.sample_prior_predictive()


start = findMAP(model)

with model:
    trace = pm.sample(**SAMPLE_KWARGS) # ,start=start, ,step = pm.Metropolis()

az.plot_trace(trace);plt.show()

showRhat(trace)
az.loo(trace)

"""
Computed from 2000 posterior samples and 48 observations log-likelihood matrix.

         Estimate       SE
elpd_loo   111.12     3.28
p_loo       20.14        -

There has been a warning during the calculation. Please check the results.
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.5]   (good)       16   33.3%
 (0.5, 0.7]   (ok)         22   45.8%
   (0.7, 1]   (bad)        10   20.8%
   (1, Inf)   (very bad)    0    0.0%
"""



summary = az.summary(trace)
summary.to_csv(save_path+'bayesian/summary3.csv', index=True,header=True)
ppc = pm.sample_posterior_predictive(trace, model=model,var_names=["likelihood",'trend',"beta_jabs","beta_cpi","beta_unknown","beta"])




fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False)
ax[0].plot(data["Month"],_sample(ppc["likelihood"], 100).T * y_max,color="blue",alpha=0.05,)
data.plot.scatter(x="Month", y="y", color="k", ax=ax[0])
ax[0].set_title("Posterior predictive")
#posterior_trend = linear_trace.posterior["trend"].stack(sample=("draw", "chain")).T   # linear_with_seasonality_trace.posterior["trend"]
posterior_trend = trace.posterior["trend"].stack(sample=("draw", "chain")).T   # linear_with_seasonality_trace.posterior["trend"]
ax[1].plot(data["Month"], _sample(posterior_trend, 100).T * y_max, color="blue", alpha=0.05)
data.plot.scatter(x="Month", y="y", color="k", ax=ax[1])
ax[1].set_title("Posterior trend lines")
posterior_seasonality = (trace.posterior["seasonality"].stack(sample=("draw", "chain")).T)
ax[2].plot(data["Month"].iloc[:12],_sample(posterior_seasonality[:, :12], 100).T * 100,color="blue",alpha=0.05,)
ax[2].set_title("Posterior seasonality")
ax[2].set_ylabel("Percent change")
formatter = mdates.DateFormatter("%b")
ax[2].xaxis.set_major_formatter(formatter);plt.show()



toshow = ["beta_jabs","beta_cpi","beta_unknown"]

beta_dict = {}
fig, ax = plt.subplots(nrows=len(toshow ), ncols=1, sharex=False)
for i,e in enumerate(toshow):
    sample = _sample(ppc[e], 100).T
    ax[i].plot(data["Month"],sample,color="blue",alpha=0.05,)
    mean = sample.mean(axis=1)
    std = sample.std(axis=1)
    beta_dict[e] = {}
    beta_dict[e]['mean'] = mean
    beta_dict[e]['std'] = std
    ax[i].plot(data["Month"],mean,color="black",alpha=0.7,label='mean')
    ax[i].plot(data["Month"],mean-std,color="red",alpha=0.7,label='lower')
    ax[i].plot(data["Month"],mean+std,color="red",alpha=0.7,label='upper')
    ax[i].set_title(e)


formatter = mdates.DateFormatter("%b")
ax[i].xaxis.set_major_formatter(formatter);plt.show()

graph = pm.model_to_graphviz(model)
graph.view()


df = data.copy()
confInt = []

for k,v in beta_dict.items():
    print(k,v)
    plt.plot(data["Month"],v['mean'],label=k)
    df[k]= v['mean']
    df[k+'_lower']= v['mean']-v['std']
    df[k+'_upper']= v['mean']+v['std']
    confInt.append([k+'_lower',k+'_upper'])
    

plt.legend()
plt.show()


showDTplot(df,col2show=toshow,offset=0,title='betas: factors influencing births analysed by a bayesian model',confInt=confInt,target='',horizon=12,year=False,showBorder=False,plotname='')

for i,b in enumerate(toshow):
    showDTplot(df,col2show=[b],offset=0,title='beta: ' +b+' ,factors influencing births analysed by a bayesian model',ylabel='Value',confInt=[confInt[i]],target='',horizon=12,year=False,showBorder=False,plotname='')

