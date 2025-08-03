import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# 设置页面布局
st.set_page_config(layout="wide", page_title="港股REITs配对交易平台", page_icon="📊")
st.title("📊 港股REITs配对交易量化平台")
st.write("本平台用于发现港股REITs之间的配对交易机会，基于相对价格关系的偏移进行交易")

# 获取港股REITs列表
hk_reits = {
    "领展房产基金": "0823.HK",
    "冠君产业信托": "2778.HK",
    "置富产业信托": "0778.HK",
    "泓富产业信托": "0808.HK",
    "越秀房产信托": "0405.HK",
    "阳光房地产基金": "0435.HK",
    "富豪产业信托": "1881.HK",
    "开元产业信托": "1275.HK",
    "春泉产业信托": "1426.HK",
    "招商局商业房托": "1503.HK",
    "顺丰房托": "2191.HK"
}

# 侧边栏控制面板
st.sidebar.header("交易参数设置")
start_date = st.sidebar.date_input("开始日期", datetime.today() - timedelta(days=365*3))
end_date = st.sidebar.date_input("结束日期", datetime.today())
reit1 = st.sidebar.selectbox("选择第一只REIT", list(hk_reits.keys()), index=0)
reit2 = st.sidebar.selectbox("选择第二只REIT", list(hk_reits.keys()), index=1)
z_score_threshold = st.sidebar.slider("Z-score阈值", 1.0, 3.0, 2.0, 0.1)
lookback = st.sidebar.slider("回看窗口(天)", 30, 365, 90, 10)
initial_capital = st.sidebar.number_input("初始资金(HKD)", 10000, 1000000, 100000, 10000)
trade_size = st.sidebar.number_input("每笔交易数量", 100, 10000, 1000, 100)

# 获取数据
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data['Adj Close']

ticker1 = hk_reits[reit1]
ticker2 = hk_reits[reit2]

try:
    with st.spinner('正在加载数据...'):
        data1 = load_data(ticker1, start_date, end_date)
        data2 = load_data(ticker2, start_date, end_date)
    
    # 合并数据
    df = pd.DataFrame({reit1: data1, reit2: data2})
    df = df.dropna()
    
    # 计算对数价格
    df[f'{reit1}_log'] = np.log(df[reit1])
    df[f'{reit2}_log'] = np.log(df[reit2])
    
    # 协整检验
    score, pvalue, _ = coint(df[reit1], df[reit2])
    
    # 计算对冲比率
    model = LinearRegression()
    X = df[reit1].values.reshape(-1, 1)
    y = df[reit2].values
    model.fit(X, y)
    hedge_ratio = model.coef_[0]
    
    # 计算价差
    df['spread'] = df[reit2] - hedge_ratio * df[reit1]
    
    # 计算Z-score
    mean_spread = df['spread'].rolling(window=lookback).mean()
    std_spread = df['spread'].rolling(window=lookback).std()
    df['z_score'] = (df['spread'] - mean_spread) / std_spread
    
    # 生成交易信号
    df['long_signal'] = np.where(df['z_score'] < -z_score_threshold, 1, 0)
    df['short_signal'] = np.where(df['z_score'] > z_score_threshold, 1, 0)
    df['exit_signal'] = np.where(np.abs(df['z_score']) < 0.5, 1, 0)
    
    # 模拟交易
    positions = pd.DataFrame(index=df.index)
    positions[reit1] = 0
    positions[reit2] = 0
    positions['cash'] = initial_capital
    positions['total'] = initial_capital
    positions['in_position'] = False
    
    for i in range(1, len(df)):
        prev_position = positions.iloc[i-1].copy()
        current = positions.iloc[i].copy()
        
        # 如果没有持仓
        if not prev_position['in_position']:
            if df.iloc[i]['long_signal']:  # 做多价差 (买入REIT2，卖出REIT1)
                current[reit1] = -trade_size
                current[reit2] = trade_size
                current['cash'] = prev_position['cash'] - (df.iloc[i][reit2] * trade_size) + (df.iloc[i][reit1] * trade_size)
                current['in_position'] = True
            elif df.iloc[i]['short_signal']:  # 做空价差 (买入REIT1，卖出REIT2)
                current[reit1] = trade_size
                current[reit2] = -trade_size
                current['cash'] = prev_position['cash'] - (df.iloc[i][reit1] * trade_size) + (df.iloc[i][reit2] * trade_size)
                current['in_position'] = True
            else:
                current = prev_position
        else:
            if df.iloc[i]['exit_signal']:  # 平仓
                # 计算平仓价值
                pnl_reit1 = (df.iloc[i][reit1] - df.iloc[i-1][reit1]) * prev_position[reit1]
                pnl_reit2 = (df.iloc[i][reit2] - df.iloc[i-1][reit2]) * prev_position[reit2]
                
                current['cash'] = prev_position['cash'] + pnl_reit1 + pnl_reit2
                current[reit1] = 0
                current[reit2] = 0
                current['in_position'] = False
            else:
                # 更新持仓价值
                current[reit1] = prev_position[reit1]
                current[reit2] = prev_position[reit2]
                current['in_position'] = True
        
        # 计算总资产
        current['total'] = current['cash'] + (current[reit1] * df.iloc[i][reit1]) + (current[reit2] * df.iloc[i][reit2])
        positions.iloc[i] = current
    
    # 计算每日回报
    positions['returns'] = positions['total'].pct_change()
    
    # 计算性能指标
    total_return = (positions['total'].iloc[-1] / initial_capital - 1) * 100
    annualized_return = (positions['total'].iloc[-1] / initial_capital) ** (252/len(df)) - 1
    annualized_vol = positions['returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = (positions['total'] / positions['total'].cummax() - 1).min() * 100
    
    # 显示结果
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总收益率", f"{total_return:.2f}%")
    col2.metric("年化收益率", f"{annualized_return*100:.2f}%")
    col3.metric("夏普比率", f"{sharpe_ratio:.2f}")
    col4.metric("最大回撤", f"{max_drawdown:.2f}%")
    
    # 显示协整检验结果
    st.subheader("协整检验结果")
    st.write(f"协整检验p值: {pvalue:.4f} {'(协整关系显著)' if pvalue < 0.05 else '(无显著协整关系)'}")
    st.write(f"对冲比率: {hedge_ratio:.4f} (每买入1单位{reit1}，需卖出{hedge_ratio:.4f}单位{reit2}进行对冲)")
    
    # 创建图表
    fig = go.Figure()
    
    # 价格图表
    fig.add_trace(go.Scatter(x=df.index, y=df[reit1], name=reit1, line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df.index, y=df[reit2], name=reit2, line=dict(color='#ff7f0e')))
    
    # Z-score图表
    fig.add_trace(go.Scatter(x=df.index, y=df['z_score'], name='Z-Score', 
                             line=dict(color='#2ca02c'), yaxis='y2'))
    
    # 添加阈值线
    fig.add_hline(y=z_score_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"上阈值: {z_score_threshold}", annotation_position="top right", yaxis='y2')
    fig.add_hline(y=-z_score_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"下阈值: {-z_score_threshold}", annotation_position="bottom right", yaxis='y2')
    fig.add_hline(y=0, line_dash="dash", line_color="grey", yaxis='y2')
    
    # 添加信号点
    long_signals = df[df['long_signal'] == 1]
    short_signals = df[df['short_signal'] == 1]
    
    fig.add_trace(go.Scatter(x=long_signals.index, y=long_signals['z_score'], 
                             mode='markers', name='买入信号', marker=dict(color='green', size=8), yaxis='y2'))
    fig.add_trace(go.Scatter(x=short_signals.index, y=short_signals['z_score'], 
                             mode='markers', name='卖出信号', marker=dict(color='red', size=8), yaxis='y2'))
    
    # 设置图表布局
    fig.update_layout(
        title=f'{reit1} 与 {reit2} 配对交易分析',
        xaxis_title='日期',
        yaxis_title='价格 (HKD)',
        yaxis2=dict(title='Z-Score', overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 资产曲线
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=positions.index, y=positions['total'], name='总资产', line=dict(color='#9467bd')))
    fig2.update_layout(
        title='投资组合表现',
        xaxis_title='日期',
        yaxis_title='资产价值 (HKD)',
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 显示数据
    st.subheader("最新数据")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{reit1}**")
        st.write(f"当前价格: {df[reit1].iloc[-1]:.2f} HKD")
        st.write(f"3年收益率: {(df[reit1].iloc[-1]/df[reit1].iloc[0]-1)*100:.2f}%")
    
    with col2:
        st.write(f"**{reit2}**")
        st.write(f"当前价格: {df[reit2].iloc[-1]:.2f} HKD")
        st.write(f"3年收益率: {(df[reit2].iloc[-1]/df[reit2].iloc[0]-1)*100:.2f}%")
    
    st.write(f"当前价差: {df['spread'].iloc[-1]:.2f}, Z-Score: {df['z_score'].iloc[-1]:.2f}")
    
    # 显示持仓详情
    st.subheader("交易记录")
    st.dataframe(positions.tail(10))
    
    # 相关性分析
    st.subheader("相关性分析")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=reit1, y=reit2, data=df, ax=ax)
        plt.title(f'{reit1} vs {reit2} 价格关系')
        plt.xlabel(reit1)
        plt.ylabel(reit2)
        st.pyplot(fig_corr)
    
    with col2:
        st.write(f"相关系数: {df[reit1].corr(df[reit2]):.4f}")
        st.write(f"协整检验p值: {pvalue:.6f}")
        st.write(f"对冲比率: {hedge_ratio:.4f}")
        st.write(f"平均价差: {df['spread'].mean():.2f}")
        st.write(f"价差标准差: {df['spread'].std():.2f}")
    
except Exception as e:
    st.error(f"数据加载错误: {str(e)}")
    st.info("请确保选择的REITs在指定日期范围内有可用数据")

# 策略说明
st.sidebar.markdown("### 策略说明")
st.sidebar.markdown("""
**配对交易策略**:
1. 选择两只高度相关的港股REITs
2. 计算它们的价差和对冲比率
3. 当价差的Z-score超过阈值时开仓:
   - Z-score > 上阈值: 做空价差(买入REIT1，卖出REIT2)
   - Z-score < 下阈值: 做多价差(买入REIT2，卖出REIT1)
4. 当Z-score回归到0附近时平仓
""")