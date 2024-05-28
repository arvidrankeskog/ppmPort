# %%
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from precise.skaters.covariance.ewapmfactory import ewa_pm_factory
from precise.skaters.portfoliostatic.ppoport import ppo_vol_long_port, ppo_sharpe_long_port # <- portfolio skater
from precise.skaters.portfoliostatic.equalport import equal_long_port # <- portfolio skater
from precise.skaters.covarianceutil.datafunctions import cov_to_corrcoef
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

class ppm:
    # Init!...
    def __init__(self, n_obs:int):
        self.n_obs = n_obs # Market days of historical values used in the estimation of the covariance matrix
            
    # Filter - min days
    def filter_min_days(self, df_mfPrices:pd.DataFrame, min_obs:int):
        n = df_mfPrices[['id']].groupby(by='id', sort=False).size()
        idx = n.index[n>min_obs]
        mask = df_mfPrices['id'].isin(idx)

        return df_mfPrices.loc[mask,:].copy()

    # Dataframe with log returns.  
    def mfLogRet(self, df_mfPrices:pd.DataFrame):        
        list_prices = []
        list_tooNew = []
        ids = df_mfPrices['id'].unique()
        for id in ids:
            mask = (df_mfPrices['id'] == id)
            df_id = df_mfPrices.loc[mask, ['s']]
            if (df_id.shape[0] > (int(self.n_obs)+1)):
                p = df_id.values[(-int(self.n_obs)-1):, 0:1] # We use sell prices
                p_logRet = np.diff(np.log(p), axis=0)
                list_prices.append(p_logRet)
            else:
                list_tooNew.append(id)
        ids2 = ids[~np.isin(ids, list_tooNew)]
        nd_logRet = np.concatenate(list_prices, axis=1)
        df_logRet = pd.DataFrame(nd_logRet, columns=ids2)

        return df_logRet   
    
    # Estimation of covariance matrix
    #   Using the precise package written by Peter Cotton 
    def cov_est(self, df_logRet:pd.DataFrame):
        s = {}
        for y in df_logRet.values:
            x, x_cov, s = ewa_pm_factory(s=s, y=y, k=1, r=0.01, n_emp=self.n_obs, target=0)
        df_cov_est =  pd.DataFrame(index=df_logRet.columns, columns=df_logRet.columns, data=x_cov)

        return df_cov_est
    
    # Adjust the diagonal of the correlation matrix
    def fee_adjusted_diag(self, df_cov_est:pd.DataFrame, list_weights:list):
        n = df_cov_est.shape[0]
        if (len(list_weights) == n):
            weights = np.array(list_weights)
        else:
            weights = np.array([1]*n)
        cov_est = df_cov_est.values
        cov_est[range(n), range(n)] = cov_est[range(n), range(n)]*weights
        df_cov_est = pd.DataFrame(index=df_cov_est.index, columns=df_cov_est.index, data=cov_est)
        
        return df_cov_est

    # Using the precise package written by Peter Cotton 
    #   Based on PyPortfolioOpt
    def portfolio(self, method: str, df_mfPrices:pd.DataFrame, w_cov:int, list_weights = []):
        df_logRet = self.mfLogRet(df_mfPrices)
        df_cov = self.cov_est(df_logRet)*252 # Annualized
        if len(list_weights) > 0:
            df_cov = self.fee_adjusted_diag(df_cov, list_weights)
        corr = cov_to_corrcoef(df_cov.values)
        if method == 'max_sharpe':
            w = ppo_sharpe_long_port(cov=df_cov.values)
        elif method == 'min_vol':
            w = ppo_vol_long_port(cov=df_cov.values)
        elif method == 'min_vol_wCorr':
            w = ppo_vol_long_port(cov=corr)
        elif method == 'equal':
            w = equal_long_port(cov=df_cov.values)
        else:
            st.write('Not a valid portfolio method!')

        return w

##########################################
# Some stuff used in the streamlit app:) #
##########################################

# Value in sell price
def ts_portfolio_value(df_mfPrices:pd.DataFrame, port_shares_old:np.ndarray, port_shares_new:np.ndarray, start_date:pd.Timestamp, end_date:pd.Timestamp, trans_costs:float, portfolio:int):
    list_fv = []
    list_sell_price_0 = []
    list_buy_price_0 = []

    ids = df_mfPrices['id'].unique()
    for id in ids:
        mask = (df_mfPrices['id'] == id) & (df_mfPrices['d'] > start_date) & (df_mfPrices['d'] <= end_date)
        df_fv = df_mfPrices.loc[mask, ['s', 'b']]

        df_s = pd.DataFrame(df_fv[['s']].values, index=df_mfPrices.loc[mask, 'd'].values, columns=['s']) 
        list_fv.append(df_s)

        list_sell_price_0.append(df_fv.loc[df_fv.index[0], 's'])
        list_buy_price_0.append(df_fv.loc[df_fv.index[0], 'b'])
    
    fv = pd.concat(list_fv, axis=1, join='inner').values
    ts_port_fv = np.sum(fv @ port_shares_new.T, axis=1)  
    
    # Cost of spread
    temp = (port_shares_new[portfolio,:]-port_shares_old[portfolio,:])*(np.array(list_buy_price_0)-np.array(list_sell_price_0))
    mask = (temp > 0)
    if len(temp[mask]) > 0:
        ts_port_fv[0] = ts_port_fv[0] - np.sum(temp[mask])
        trans_costs += np.sum(temp[mask])

    return ts_port_fv, trans_costs

# Walkforward backtest with rebalancing accoring to https://blog.thinknewfound.com/2019/07/timing-luck-and-systematic-value/ 
# Rebalance frequency in months
def backtest(port_method: str, df_mfPrices:pd.DataFrame, list_ids:list, r_freq_months=3, w_cov = 100, min_obs = 1*252):
    max_date = str(max(df_mfPrices['d']))[0:10] # Latest date of data in db
    p = ppm(w_cov)
    switch = 0
    valid_ids = np.array(list_ids)[np.isin(np.array(list_ids), df_mfPrices['id'].values)]
    if valid_ids.shape[0] != len(list_ids):
        not_valid_ids = np.array(list_ids)[~np.isin(np.array(list_ids), valid_ids)]
        st.write('Incorrect ids: ', not_valid_ids)
        switch = 1
    if 12 % r_freq_months != 0:
        st.write('The remainder of 12 % r_freq_months has to be 0!')
        switch = 1
    if switch == 0:
        df_mfPrices_sub = p.filter_min_days(df_mfPrices, min_obs)
        mask = df_mfPrices_sub['id'].isin(list_ids)
        df_mfPrices_sub = df_mfPrices_sub.loc[mask, :]

        # Fund shares
        #   Initialize with equal sized portfolio weights
        #   After init_backtest is this array used for fund shares and not portfolio weights
        port_shares = np.full((int(12/r_freq_months), len(list_ids)), 1/len(list_ids))
        port_shares_new = np.full((int(12/r_freq_months), len(list_ids)), 1/len(list_ids))
        
        df_start_date = df_mfPrices_sub[['id', 'd']].groupby(by='id', sort=False).first()
        min_date = df_start_date['d'].max()
        # Start backtest after w_cov market days
        date_i = min_date + pd.Timedelta(days=w_cov*np.ceil(7/5))
        dates_backtest = df_mfPrices_sub.loc[df_mfPrices_sub['d'] >= date_i, 'd'].values
        i_port = 0
        init_backtest = 0
        trans_costs = 0
        list_fv = []
        while date_i <= pd.to_datetime(max_date):
            date_next = date_i + pd.Timedelta(days=int(365/(port_shares.shape[0])))
            mask = (df_mfPrices_sub['d'] <= date_i)
            df_mfPrices_sub_i = df_mfPrices_sub.loc[mask, :]
            df_sell_price_i = df_mfPrices_sub_i[['id','s']].groupby(by='id', sort=False).last()
            sell_price_i = (df_sell_price_i.values).flatten()
            
            port_shares = port_shares_new.copy()
            w_port = p.portfolio(port_method, df_mfPrices_sub_i, w_cov)
            
            if init_backtest == 0:
                # We start with 1 SEK in total
                temp = w_port*(1/round(12/r_freq_months,0))/sell_price_i   
                temp = np.reshape(temp, (1, temp.shape[0]))
                port_shares = np.repeat(temp, round(12/r_freq_months,0), axis=0)
                port_shares_new = np.repeat(temp, round(12/r_freq_months,0), axis=0)
                init_backtest = 1
            else:
                port_shares_new[i_port,:] = w_port*np.sum(sell_price_i*port_shares[i_port,:])/sell_price_i   
            
            ts_port_fv, trans_costs = ts_portfolio_value(df_mfPrices_sub, port_shares, port_shares_new, date_i, date_next, trans_costs, i_port)
            
            if init_backtest == 0: 
                trans_costs = 0
            
            list_fv.append(ts_port_fv)

            if i_port < (int(12/r_freq_months)-1):
                i_port += 1
            else:
                i_port = 0                
            date_i = date_next

        fv = np.concatenate(list_fv)
        
        return fv, dates_backtest

# Hierarchical tree graph
def hier_tree_plot(np_corr, dist_linkage, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    ax2.imshow(np_corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    _ = fig.tight_layout()
    
    return fig

# Hierarchical clustering viewed from above in an interactive manner using some great stuff from Lopez De Prado
#   and plotly:)    
# Window w_cov: market days
# Windows in list_w_ret: market days
# First w in the list is used to determine the colors in the treemap
def get_clustered_treemap(df_mfPrices:pd.DataFrame, df_mfInfo:pd.DataFrame, w_cov:int, list_w_ret:list, df_clusters_old = pd.DataFrame()):
    p = ppm(n_obs=w_cov)
    if (df_mfPrices.shape[0] != 0) & (df_mfInfo.shape[0] != 0):
        df_mfPrices_sub = p.filter_min_days(df_mfPrices, min_obs=max(list_w_ret + [w_cov]))
        df_mfPrices_sub['d'] = pd.to_datetime(df_mfPrices_sub['d']) 
        df_logRet = p.mfLogRet(df_mfPrices_sub)
        df_corr = df_logRet.corr()
        
        # Ensure the correlation matrix is symmetric
        np_corr = (df_corr.values + df_corr.values.T) / 2
        np.fill_diagonal(np_corr, 1)
    
        # We convert the correlation matrix to a distance matrix before performing
        np_distance_matrix = ((1 - np_corr)/2)**0.5 # From Lopez --> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595
        
        # Hierarchical clustering
        dist_linkage = hierarchy.linkage(squareform(np_distance_matrix), method='single', metric='euclidean', optimal_ordering=True)
        ids = hierarchy.leaves_list(dist_linkage)
        
        df_clusters = pd.DataFrame(index=df_logRet.columns)
        idx = df_clusters.index[ids]

        fig_h_tree = hier_tree_plot(np_corr, dist_linkage, ['']*len(idx)) # Too many indices
        # Distance levels. The distance is defined above. Single linkage in the HC. 
        v = [0.00, 0.10, 0.20, 0.40, 0.50]
        list_str_clusters = []
        for t in v:
            cluster_ids = hierarchy.fcluster(dist_linkage, t, criterion="distance")
            df_clusters.loc[idx, 'cluster_'+("%.2f" % t)] = ['C'+("%.2f" % t)+'_'+str(i) for i in cluster_ids]
            list_str_clusters.append('cluster_'+("%.2f" % t))

        latest_date = df_mfPrices_sub['d'].max()
        p_latest = df_mfPrices_sub[['id', 's']].groupby(by='id', sort=False).last()
        list_gross_ret_labels = []
        for w in list_w_ret:
            date_w = latest_date - pd.Timedelta(days=w)
            mask = (df_mfPrices_sub['d'] <= date_w)
            p_w = df_mfPrices_sub.loc[mask, ['id', 's']].groupby(by='id', sort=False).last()
            ret = p_latest.values/p_w
            df_clusters.loc[:, f'gross_ret_{w}'] = round(ret,2) # In %  
            list_gross_ret_labels.append(f'gross_ret_{w}')
        
        # Add some info from df_mfInfo
        df_clusters.reset_index(drop=False, inplace=True, names='id')
        df_clusters = df_clusters.merge(df_mfInfo[['id', 'name', 'category', 'fee_pct_gross', 'fee_pct_net', 'currency']], left_on='id', right_on='id', how='left')
        df_clusters['id_str'] = df_clusters['id'].astype(str)
    else: 
        df_clusters = df_clusters_old
    # Hover stuff
    list_hover = list_gross_ret_labels[1:] + ['id_str', 'name', 'category', 'fee_pct_gross', 'fee_pct_net', 'currency']
    fig = px.treemap(df_clusters, path=[px.Constant("All mutual funds in the database")]+ list(reversed(list_str_clusters)),
            values=list_gross_ret_labels[0], hover_data = list_hover, #(list_gross_ret_labels[1:] + ['id', 'name']) 
            color=list_gross_ret_labels[0],
            color_continuous_scale='RdBu', 
            width=900,height=1100)
    fig.update_traces(hovertemplate='%{customdata[0]}')      
    # Hover stuff
    fig.data[0].customdata = [
        [
        v[(len(list_w_ret)-1)] +', ' + # Name
        v[(len(list_w_ret))]+ '<br>'+ # Id
        f'ret_{list_w_ret[0]} = {int(round((v[(len(list_hover))] - 1)*100, 0))}%<br>'+ # Percentage return
        (''.join([(f'ret_{list_w_ret[j+1]} = {int(round((v[j] - 1)*100, 0))}%<br>') for j in range((len(list_w_ret)-1))])) +
        'Category: ' + v[(len(list_w_ret)+1)] + '<br>' + # Category
        f'Gross_fee_pct: {v[(len(list_w_ret)+2)]} <br>'+ # Fee, gross
        f'Net_fee_pct: {v[(len(list_w_ret)+3)]} <br>'+ # Fee, net
        'Currency: ' + v[(len(list_w_ret)+4)] # Currency
        ] 
        if 'C0.00' in fig.data[0].ids[i] 
        else [""]
        for i, v in enumerate(fig.data[0].customdata)       
        ]

    return df_clusters, fig, fig_h_tree