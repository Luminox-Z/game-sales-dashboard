import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ========== 页面基础配置 ==========
st.set_page_config(page_title="全球游戏销量可视化分析平台", layout="wide")
plt_colors = px.colors.qualitative.Set2

# ========== 全局原始数据仅加载一次 ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "vgsales.csv",
            sep=",",
            encoding="utf-8",
            engine="python"
        )
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    required_cols = [
        'Rank', 'Name', 'Platform', 'Year', 'Genre',
        'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
        'Other_Sales', 'Global_Sales'
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"数据列缺失：{missing}，CSV文件格式异常，请换回原版数据集")
        st.dataframe(df.head())
        st.stop()

    df = df.dropna(subset=['Year'])
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)
    df = df[(df['Year'] >= 1980) & (df['Year'] <= 2020)]
    df = df[df['Global_Sales'] > 0]

    df['NA_Ratio'] = df['NA_Sales'] / df['Global_Sales']
    df['EU_Ratio'] = df['EU_Sales'] / df['Global_Sales']
    df['JP_Ratio'] = df['JP_Sales'] / df['Global_Sales']
    df['Other_Ratio'] = df['Other_Sales'] / df['Global_Sales']

    X = df[['NA_Ratio', 'EU_Ratio', 'JP_Ratio', 'Other_Ratio']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    cluster_map = {0: '欧美主导型', 1: '日本本土型', 2: '全球通吃型'}
    df['Cluster'] = df['Cluster'].map(cluster_map)

    return df

if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = load_data()
df = st.session_state["raw_df"]

# ========== 筛选计算缓存 ==========
@st.cache_data
def filter_dataset(df, y_min, y_max, plats, genres, pub):
    tmp = df[
        (df['Year'] >= y_min) & (df['Year'] <= y_max) &
        (df['Platform'].isin(plats)) &
        (df['Genre'].isin(genres))
    ]
    if pub:
        tmp = tmp[tmp['Publisher'].str.contains(pub, case=False, na=False)]
    return tmp

# ========== 侧边栏放入表单！！拖动滑块不会实时刷新页面 ==========
with st.sidebar:
    st.header("🔍 筛选条件")
    # 所有筛选控件包裹在form内，仅点击按钮才刷新页面
    with st.form("filter_form"):
        year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
        year_range = st.slider("发行年份区间", year_min, year_max, (year_min, year_max), key="slider_year")

        platform_list = sorted(df['Platform'].unique().tolist())
        selected_platform = st.multiselect("游戏平台", platform_list, default=platform_list, key="multi_platform")

        genre_list = sorted(df['Genre'].unique().tolist())
        selected_genre = st.multiselect("游戏类型", genre_list, default=genre_list, key="multi_genre")

        publisher_input = st.text_input("发行商关键词（留空为全部）", "", key="input_publisher")

        # 确认按钮，只有点这里才更新数据、刷新页面
        submit_btn = st.form_submit_button("确认筛选")

# 初始化筛选数据
if "filtered_data" not in st.session_state:
    st.session_state["filtered_data"] = df

# 点击按钮才重新计算筛选结果
if submit_btn:
    st.session_state["filtered_data"] = filter_dataset(df, year_range[0], year_range[1], selected_platform, selected_genre, publisher_input)
filtered_df = st.session_state["filtered_data"]

# ========== 数据预览折叠框 ==========
with st.expander("📋 点击查看数据读取预览（确认数据是否正常）", expanded=False):
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"共 {len(df)} 条记录，{len(df.columns)} 个字段")

# ========== 主页面 ==========
st.title("🎮 全球电子游戏销量交互式可视化分析平台")
st.markdown("基于1980-2020年游戏销量数据，点击侧边【确认筛选】更新图表")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 数据概览", "📈 时间演化", "🎯 平台格局",
    "🎲 类型分布", "🌍 区域市场", "🔍 聚类分析"
])

# ---------------------- 缓存函数统一抽取 ----------------------
@st.cache_data
def get_top10_table(data):
    return data.sort_values('Global_Sales', ascending=False).head(10)[
        ['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']
    ].reset_index(drop=True)

@st.cache_data
def get_year_group(data):
    return data.groupby('Year').agg(
        发行数量=('Name', 'count'),
        全球总销量=('Global_Sales', 'sum')
    ).reset_index()

@st.cache_data
def get_platform_top15(data):
    return data.groupby('Platform')['Global_Sales'].sum() \
        .sort_values(ascending=False).head(15).reset_index()

@st.cache_data
def get_genre_sales_data(data):
    return data.groupby('Genre')['Global_Sales'].sum() \
        .sort_values(ascending=True).reset_index()

@st.cache_data
def get_region_pie_data(data):
    return pd.DataFrame({
        '区域': ['北美', '欧洲', '日本', '其他地区'],
        '销量（百万套）': [
            data['NA_Sales'].sum(),
            data['EU_Sales'].sum(),
            data['JP_Sales'].sum(),
            data['Other_Sales'].sum()
        ]
    })

@st.cache_data
def get_corr_matrix(data):
    corr = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].corr()
    corr.columns = ['北美', '欧洲', '日本', '其他']
    corr.index = ['北美', '欧洲', '日本', '其他']
    return corr

@st.cache_data
def get_cluster_stats(data):
    return data.groupby('Cluster').agg(
        北美占比=('NA_Ratio', 'mean'),
        欧洲占比=('EU_Ratio', 'mean'),
        日本占比=('JP_Ratio', 'mean'),
        其他占比=('Other_Ratio', 'mean'),
        游戏数量=('Name', 'count')
    ).reset_index()

@st.cache_data
def get_cluster_sample_table(data, cluster_name):
    return data[data['Cluster'] == cluster_name] \
        .sort_values('Global_Sales', ascending=False).head(20)[
        ['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']
    ].reset_index(drop=True)

# ---------------------- 标签1 数据概览 ----------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("游戏总数", f"{len(filtered_df)} 款")
    col2.metric("全球总销量", f"{filtered_df['Global_Sales'].sum():.1f} 百万套")
    col3.metric("平均单款销量", f"{filtered_df['Global_Sales'].mean():.2f} 百万套")
    col4.metric("涉及平台数", f"{filtered_df['Platform'].nunique()} 个")

    st.subheader("🏆 全球销量Top10游戏")
    top10 = get_top10_table(filtered_df)
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ---------------------- 标签2 时间演化 ----------------------
with tab2:
    st.subheader("历年游戏发行数量与全球销量趋势")
    year_stats = get_year_group(filtered_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=year_stats['Year'], y=year_stats['发行数量'],
        name='发行数量（款）', marker_color='#4C72B0', opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=year_stats['Year'], y=year_stats['全球总销量'],
        name='全球总销量（百万套）', mode='lines+markers',
        line=dict(color='#DD8452', width=3), yaxis='y2'
    ))

    fig.update_layout(
        xaxis_title='发行年份',
        yaxis=dict(title=dict(text='发行数量（款）', font=dict(color='#4C72B0'))),
        yaxis2=dict(
            title=dict(text='全球总销量（百万套）', font=dict(color='#DD8452')),
            overlaying='y', side='right'
        ),
        legend=dict(orientation='h', y=1.1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True, key="fig_year_trend")

# ---------------------- 标签3 平台格局 ----------------------
with tab3:
    st.subheader("各平台累计全球销量排行")
    platform_sales = get_platform_top15(filtered_df)

    fig = px.bar(
        platform_sales, x='Platform', y='Global_Sales',
        color='Global_Sales', color_continuous_scale='Viridis',
        labels={'Global_Sales': '累计全球销量（百万套）'},
        text_auto='.1f'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="fig_platform")

# ---------------------- 标签4 类型分布（已修复KeyError） ----------------------
with tab4:
    st.subheader("各游戏类型全球销量对比")
    genre_sales = get_genre_sales_data(filtered_df)
    # 清洗Genre空值、空白字符
    genre_sales = genre_sales.dropna(subset=["Genre"])
    genre_sales = genre_sales[genre_sales["Genre"].str.strip() != ""]
    genre_sales = genre_sales.reset_index(drop=True)

    # 替换为go底层绘图，彻底规避px分组KeyError
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=genre_sales["Genre"],
        x=genre_sales["Global_Sales"],
        orientation="h",
        text=genre_sales["Global_Sales"].round(1),
        textposition="auto",
        marker_color=plt_colors[0]
    ))
    fig.update_layout(
        xaxis_title="全球总销量（百万套）",
        yaxis_title="游戏类型",
        height=550
    )
    st.plotly_chart(fig, use_container_width=True, key="fig_genre")

# ---------------------- 标签5 区域市场 ----------------------
with tab5:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("四大区域市场总销量对比")
        region_total = get_region_pie_data(filtered_df)
        fig = px.pie(
            region_total, values='销量（百万套）', names='区域',
            color_discrete_sequence=plt_colors, hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True, key="fig_region_pie")

    with col2:
        st.subheader("区域市场销量相关性热力图")
        corr = get_corr_matrix(filtered_df)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale='RdBu_r',
            zmin=0, zmax=1, height=450
        )
        st.plotly_chart(fig, use_container_width=True, key="fig_corr_heat")

# ---------------------- 标签6 聚类分析 ----------------------
with tab6:
    st.subheader("三类游戏区域特征对比雷达图")
    cluster_stats = get_cluster_stats(filtered_df)

    categories = ['北美占比', '欧洲占比', '日本占比', '其他占比']
    fig = go.Figure()
    colors = ['#4C72B0', '#55A868', '#DD8452']
    for i, row in cluster_stats.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['北美占比'], row['欧洲占比'], row['日本占比'], row['其他占比']],
            theta=categories, fill='toself',
            name=f"{row['Cluster']}（{row['游戏数量']}款）",
            line_color=colors[i]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.6])),
        height=550, legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True, key="fig_radar_cluster")

    st.subheader("各聚类类别游戏示例")
    cluster_choice = st.selectbox("选择类别查看游戏列表", cluster_stats['Cluster'].tolist(), key="select_cluster_type")
    cluster_sample = get_cluster_sample_table(filtered_df, cluster_choice)
    st.dataframe(cluster_sample, use_container_width=True, hide_index=True)

st.divider()
st.caption("数据来源：VGChartz | 基于Streamlit构建")