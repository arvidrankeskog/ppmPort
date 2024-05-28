import streamlit as st
import pandas as pd
from ppm import ppm, backtest, get_clustered_treemap
from st_supabase_connection import SupabaseConnection
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px

page_title = "ppmOpt"
layout = "wide"
st.set_page_config(page_title=page_title, layout=layout)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

feature = option_menu(
    menu_title=None,
    options=["PPM mutual funds map", "Portfolio construction",  "Walk forward backtest - rebalancing"],
    icons=["database-down", "database-down"],  # https://icons.getbootstrap.com/
    orientation="horizontal",
)

# -------------------------------
# Fct. for init. connection to db
# -------------------------------
def session_init():
    if "conn" not in st.session_state:
        st.session_state["conn"] = st.connection("supabase", type=SupabaseConnection)

    return st.session_state["conn"]

# Get data from the database
def get_data(only_mfinfo=False):
    # Initialize connection.
    conn = session_init()

    # Query all rows from mfInfo 
    rows = conn.query("*", table="mfinfo", ttl=0).execute()
    df_mfInfo = pd.json_normalize(rows.data)

    if only_mfinfo:
        df_mfPrices = pd.DataFrame()
    else:
        # Query from mfPrices
        #   Hardcoded date=) 
        rows = conn.query("*", table="mfprices", ttl=0).gt('d', '2022-01-01').execute()
        df_mfPrices = pd.json_normalize(rows.data)
        df_mfPrices['d'] = pd.to_datetime(df_mfPrices['d'])
        df_mfPrices.sort_values(by='d', ascending=True, inplace=True)

    return df_mfPrices, df_mfInfo

# Treemap
def treemap(df_mfPrices:pd.DataFrame, df_mfInfo:pd.DataFrame, w_corr:int, list_w_ret:list, df_cl:pd.DataFrame):
    df_clusters, fig, fig2 = get_clustered_treemap(df_mfPrices, df_mfInfo, w_corr, list_w_ret, df_cl)
    
    return df_clusters, fig, fig2

conn = session_init()

# ----------------
# Mutual Funds map
# ----------------
if feature == 'PPM mutual funds map':
    w_corr = st.number_input("Insert the window of the sample correlation matrix (market days)", value=100)
    w_corr = int(w_corr)

    ##################
    # Return windows
    ##################

    # Init. window of return labels
    if "list_w_ret" not in st.session_state: 
        st.session_state['list_w_ret'] = [100,250,500]
        st.write('Default windows of returns: ', [100,250,500])

    # Input w_ret
    col1, col2 = st.columns(2)
    with col1: 
        w_ret = st.number_input("Insert window of the return label", value=100)
    with col2: 
        # Delete all windows of return labels
        ans_empty = st.button('Empty list of the windows of the return labels')
        if ans_empty:
            st.session_state['list_w_ret'] = []
            st.write('Windows in the ret_list', st.session_state['list_w_ret'])
        # Or add return label to list
        ans_add = st.button('Add the window the return label')
        if ans_add:
            list_w_ret = st.session_state['list_w_ret']
            if w_ret not in list_w_ret:
                list_w_ret.append(w_ret)
                st.session_state['list_w_ret'] = list_w_ret
                st.write('Windows in the ret_list', st.session_state['list_w_ret'])
            else:
                st.write(w_ret, 'is already in the list')

    ##################
    # The treemap!
    ##################

    ans_tm = st.button('Get treemap of the ppm mutual funds!')
    st.write("The first window in ret_list determines the colors in the treemap")

    if ans_tm:
        if "df_clusters" not in st.session_state:
            st.session_state["df_clusters"] = pd.DataFrame()
        # If no previous df_clusters or changed return labels
        df_cl = st.session_state["df_clusters"]
        if (df_cl.shape[0] == 0) | len([i for i in st.session_state['list_w_ret'] if i in df_cl.columns]) != len(st.session_state['list_w_ret']):
            df_mfPrices, df_mfInfo = get_data()
        else:
            df_mfPrices = df_mfInfo = pd.DataFrame()

        df_clusters, fig, fig2 = treemap(df_mfPrices, df_mfInfo, w_corr, st.session_state['list_w_ret'], df_cl)
        st.session_state['tree_plot'] = fig
        st.session_state['hier_plot'] = fig2

        df_mfPrices = df_mfInfo = pd.DataFrame()

        st.session_state["df_clusters"] = df_clusters

        st.plotly_chart(fig, width=900,height=1100, theme='streamlit')

    ans_htg = st.button('Get classical hierarchical tree graph!')
    if ans_htg:
        if 'hier_plot' in st.session_state:
            # We plot the treemap and the hierarchical tree graph
            st.plotly_chart(st.session_state['tree_plot'], width=900,height=1100, theme='streamlit')
            st.pyplot(st.session_state['hier_plot'])
        else:
            st.write('Get treemap first')
# ----------------------
# Portfolio construction
# ----------------------
elif feature == "Portfolio construction":
    if 'df_data' not in st.session_state:
        st.session_state['df_data'] = pd.DataFrame()
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        w_cov_input = st.number_input("Insert window of the data to use in the covariance estimation", value=200)
        w_cov_input = int(w_cov_input)
    with col2:
        min_obs_input = st.number_input("Insert minimum number of observations", value=252)
    with col3:
        ans_vals = st.button('Use the input values.')
        ans_def = st.button('Use the default values. 200 market days for cov. est. and 252 min. obs.')
    if ans_def:
        st.session_state['w_cov'] = 200
        st.write("w_cov = ", st.session_state['w_cov'])
        st.session_state['min_obs'] = 252
        st.write("min_obs = ", st.session_state['min_obs'])
    if ans_vals:
        st.session_state['w_cov'] = w_cov_input
        st.write("w_cov = ", w_cov_input)
        st.session_state['min_obs'] = min_obs_input
        st.write("min_obs = ", min_obs_input)
    
    # Get data
    _, df_mfInfo = get_data(only_mfinfo=True)
    funds = st.multiselect('Select mutual funds', df_mfInfo['name'])
    if len(funds) > 5:
        st.write('Max 5 mutual funds!')
    else:
        mask = df_mfInfo['name'].isin(funds)
        ids = df_mfInfo.loc[mask,'id'].to_list()
        ans_cont = st.button('Proceed!')
        list_dfs = []
        if ans_cont:
            # Get data
            list_figs = []
            for i in range(len(funds)):
                id = ids[i]
                fig = go.Figure()
                conn = session_init()
                rows = conn.query("*", table="mfprices", ttl=0).eq(column='id', value=id).execute()
                df_mfPrices = pd.json_normalize(rows.data)
                df_mfPrices['d'] = pd.to_datetime(df_mfPrices['d'])
                df_mfPrices.sort_values(by='d', ascending=True, inplace=True)                
                list_dfs.append(df_mfPrices)
                # Add to fig.
                fig.add_trace(go.Scatter(
                x=df_mfPrices['d'],
                y=df_mfPrices['s'],
                line_color='rgb(57, 255, 20)',
                name=id))
                fig.update_layout(title=go.layout.Title(text=funds[i]))
                list_figs.append(fig)
            st.session_state['figs'] = list_figs
            df = pd.concat(list_dfs) 
            st.session_state['df_data'] = df
        
        if st.session_state['df_data'].shape[0] > 0:
            # Always plot prices
            list_prices_figs = st.session_state['figs']
            for fig in list_prices_figs:
                st.plotly_chart(fig, theme="streamlit")
            # Portfolio construction
            p = ppm(n_obs=st.session_state['w_cov'])
            # If too new mutual funds
            df_sub = p.filter_min_days(st.session_state['df_data'], st.session_state['min_obs'])
            mask = df_mfInfo['id'].isin(df_sub['id'])
            names = df_mfInfo.loc[mask, 'name'].values
            # Manual input
            st.write('Scaling factors of the variances, default = 1')
            st.write('The order is important!')
            st.write(names)
            if 'list_factors' not in st.session_state:
                st.session_state['list_factors'] = [1]*names.shape[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                f = st.number_input("Insert factor", value=1)
            
            with col2:
                # Add weight (variance) label
                ans_add = st.button('Add factor')
                if ans_add:
                    list_factors = st.session_state['list_factors']
                    list_factors.append(f)
                    st.session_state['list_factors'] = list_factors
                    st.write('Scaling factors', st.session_state['list_factors'])
                # Delete windows of return labels
                ans_del = st.button('Empty list of scaling factors')
                if ans_del:
                    st.session_state['list_factors'] = []
                    st.write('Scaling factors', st.session_state['list_factors'])

            with col3:
                ans_def = st.button('Use default factors and get portfolio weights!')
                if ans_def:
                    st.session_state['list_factors'] = [1]*names.shape[0]
                ans_inp = st.button('Use input factors and get portfolio weights!')

            # Check 
            if len(st.session_state['list_factors']) != names.shape[0]:
                st.write('Incorrect number of factors!')
            
            # Get portfolio weights!
            if (ans_def | ans_inp) & (len(st.session_state['list_factors']) == names.shape[0]):
                # Get weights
                list_port_methods = ['max_sharpe', 'min_vol', 'min_vol_wCorr', 'equal']
                list_ports = []
                for i in range(len(list_port_methods)):
                    method = list_port_methods[i]
                    w_port = p.portfolio(method, df_sub, st.session_state['w_cov'])
                    list_w = []
                    for i in range(w_port.shape[0]):
                        list_w.append(round(w_port[i], 2))
                    list_ports.append(list_w)
                df = pd.DataFrame(list_ports, index=list_port_methods, columns = names)
                fig = px.bar(df, labels=['Portfolios', 'Weights'])
                fig.update_layout(
                    xaxis_title='Portfolios',
                    yaxis_title='Weights'  
                    )
                st.plotly_chart(fig, theme="streamlit")
            
                # URLs
                st.write("Cov-implied returns in the max-sharpe method. Read more about it in the PyPortfolioDocs:") 
                st.page_link("https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html", label='PyPortfolioOpt')
                st.write("Peter Cotton's package Precice:")
                st.page_link("https://github.com/microprediction/precise/tree/main", label='Precise')
# ----------------------
# WF-backtest...
# ----------------------
elif feature == "Walk forward backtest - rebalancing":
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        w_cov_input = st.number_input("Insert window of the data to use in the covariance estimation", value=200)
        w_cov_input = int(w_cov_input)
    with col2:
        min_obs_input = st.number_input("Insert minimum number of observations", value=252)
        min_obs_input = int(min_obs_input)
    with col3:
        ans_inp = st.button('Use the input values.')
        ans_def = st.button('Use the default values. 200 market days for cov. est. and 252 min. obs.')
    if ans_def:
        st.session_state['w_cov'] = 200
        st.write("w_cov = ", st.session_state['w_cov'])
        st.session_state['min_obs'] = 252
        st.write("min_obs = ", st.session_state['min_obs'])
    if ans_inp:
        st.session_state['w_cov'] = w_cov_input
        st.write("w_cov = ", w_cov_input)
        st.session_state['min_obs'] = min_obs_input
        st.write("min_obs = ", min_obs_input)
    # Get available mutual funds
    _, df_mfInfo = get_data(only_mfinfo=True)
    funds = st.multiselect('Select mutual funds', df_mfInfo['name'])
    # If PPM-portfolio
    if len(funds) > 5:
        st.write('Max 5 mutual funds in the ppm portfolio!')
    # Continue either way
    if funds:
        mask = df_mfInfo['name'].isin(funds)
        ids = df_mfInfo.loc[mask,'id'].to_list()
        names_ordered = df_mfInfo.loc[mask,'name'].to_list()
        ans_cont = st.button('Proceed!')
        list_dfs = []
        if ans_cont:
            # Get data
            for i in range(len(funds)):
                id = ids[i]
                fig = go.Figure()
                conn = session_init()
                rows = conn.query("*", table="mfprices", ttl=0).eq(column='id', value=id).execute()
                df_mfPrices = pd.json_normalize(rows.data)
                df_mfPrices['d'] = pd.to_datetime(df_mfPrices['d'])
                df_mfPrices.sort_values(by='d', ascending=True, inplace=True)                
                list_dfs.append(df_mfPrices)
                # Add to fig.
                fig.add_trace(go.Scatter(
                x=df_mfPrices['d'],
                y=df_mfPrices['s'],
                line_color='rgb(57, 255, 20)',
                name=id))
                fig.update_layout(title=go.layout.Title(text=names_ordered[i]))
                st.plotly_chart(fig, theme="streamlit")
            df = pd.concat(list_dfs) 
    
            # Backtest - walk forward...
            fig = go.Figure()
            list_port_methods = ['max_sharpe', 'min_vol', 'min_vol_wCorr', 'equal']
            for method in list_port_methods:
                fv, list_date = backtest(method, df, ids, w_cov=st.session_state['w_cov'], min_obs=st.session_state['min_obs'])
                fig.add_trace(go.Scatter(
                x=list_date,
                y=fv,
                name=method))
            fig.update_layout(title=go.layout.Title(text='Portfolio value'))
            st.plotly_chart(fig, theme="streamlit")

             # URLs
            st.write("Cov-implied returns in the max-sharpe method. Read more about it in the PyPortfolioDocs:") 
            st.page_link("https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html", label='PyPortfolioOpt')
            st.write("Peter Cotton's package Precice:")
            st.page_link("https://github.com/microprediction/precise/tree/main", label='Precise')