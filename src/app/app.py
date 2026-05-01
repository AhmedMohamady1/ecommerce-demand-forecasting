import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
import pickle
import boto3
import sys
from dotenv import load_dotenv

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

load_dotenv(os.path.join(proj_root, ".env"))

st.set_page_config(page_title="Ecommerce Demand Forecasting App", layout="wide")


@st.cache_resource
def load_prophet_models():
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    
    obj = s3_client.get_object(Bucket="ecommerce-lake", Key="gold/models/prophet/prophet_models.pkl")
    return pickle.loads(obj['Body'].read())

def generate_predictions(df):
    from config.spark_config import get_spark
    from src.preprocessing.cleaner import clean_and_enrich
    from src.preprocessing.aggregator import aggregate_to_weekly
    from src.models.prophet_model import _week_to_date

    spark = get_spark("StreamlitMonitor")
    
    df = df.copy()
    # Cast date to Python date objects for PySpark DateType compatibility
    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.date
    
    # If the user uploads a test set without sales, inject dummy sales 
    # so cleaner.py (which expects 'sales' to filter >= 0) doesn't fail.
    if 'sales' not in df.columns:
        df['sales'] = 0
        
    spark_df = spark.createDataFrame(df)
    
    # --- REUSING IDENTICAL PIPELINE PREPROCESSING ---
    clean_df = clean_and_enrich(spark_df, spark)
    weekly_df = aggregate_to_weekly(clean_df)
    
    # Convert back to Pandas for Prophet inference
    weekly_pdf = weekly_df.toPandas()
    
    test_weeks = weekly_pdf[['store', 'item', 'year', 'week_of_year']].drop_duplicates()
    test_weeks = test_weeks.rename(columns={'week_of_year': 'week'})
    
    models = load_prophet_models()
    
    predictions = []

    grouped = test_weeks.groupby(['store', 'item'])
    progress_bar = st.progress(0)
    total = len(grouped)
    
    for idx, ((s, i), grp) in enumerate(grouped):
        model = models.get((s, i))
        if model is not None:
            test_prophet = pd.DataFrame({
                "ds": [_week_to_date(r["year"], r["week"]) for _, r in grp.iterrows()],
            })
            forecast = model.predict(test_prophet)
            
            grp = grp.copy()
            grp['sales_prediction'] = forecast['yhat'].values
            predictions.append(grp)
            
        progress_bar.progress((idx + 1) / total)
        
    progress_bar.empty()
    return pd.concat(predictions, ignore_index=True)


def filter_predictions(df, year=None, stores=None, items=None, week_range=None):
    out = df.copy()
    if year is not None and 'year' in out.columns:
        out = out[out['year'] == year]
    if stores is not None and len(stores) > 0 and 'store' in out.columns:
        out = out[out['store'].isin(stores)]
    if items is not None and len(items) > 0 and 'item' in out.columns:
        out = out[out['item'].isin(items)]
    if week_range is not None and 'week' in out.columns:
        out = out[(out['week'] >= week_range[0]) & (out['week'] <= week_range[1])]
    return out


# --- Session state and navigation helpers ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

def go_to(page_name: str):
    st.session_state.page = page_name
    rerun = getattr(st, 'experimental_rerun', None)
    if callable(rerun):
        try:
            rerun()
        except Exception:
            pass


def predictions_page_name():
    df = st.session_state.get('predictions_df')
    if df is None or (hasattr(df, 'empty') and df.empty):
        return 'Year Predictions'
    if 'year' in df.columns:
        yrs = sorted([y for y in df['year'].dropna().unique().tolist()]) if not df.empty else []
        if yrs:
            return f'Year {yrs[0]} Sales Predictions'
    return 'Year Predictions'


# --- Top header + page-specific right-aligned buttons ---
col1, col2, col3 = st.columns([7, 1, 2.5], vertical_alignment="bottom")
title_ph = col1.empty()
btn2_ph = col2.empty()
btn3_ph = col3.empty()

page = st.session_state.get('page', 'Home')

home_top_clicked = False
team_top_clicked = False
viz_top_clicked = False
back_top_clicked = False
home_top_viz_clicked = False
home_top_team_clicked = False

if page == 'Home':
    team_top_clicked = btn3_ph.button('👥 Team', key='team_top')
elif page.startswith('Year'):
    right_container = btn3_ph.container()
    b1, b2 = right_container.columns([1, 1])
    home_top_clicked = b1.button('🏠 Home', key='home_top')
    viz_top_clicked = b2.button('📊 Visualization', key='viz_top')
elif page == 'Visualization':
    right_container = btn3_ph.container()
    b1, b2 = right_container.columns([1, 1])
    home_top_viz_clicked = b1.button('🏠 Home', key='home_top_viz')
    back_top_clicked = b2.button('🔙 Back to Predictions', key='back_top')
elif page == 'Team':
    home_top_team_clicked = btn3_ph.button('🏠 Home', key='home_top_team')

title_ph.title('Ecommerce Demand Forecasting App')

new_page = None
if team_top_clicked:
    new_page = 'Team'
elif home_top_clicked or home_top_viz_clicked or home_top_team_clicked:
    new_page = 'Home'
elif viz_top_clicked:
    new_page = 'Visualization'
elif back_top_clicked:
    new_page = predictions_page_name()

if new_page is not None:
    st.session_state.page = new_page
    page = new_page
    title_ph.empty()
    btn2_ph.empty()
    btn3_ph.empty()
    title_ph.title('Ecommerce Demand Forecasting App')
    if page == 'Home':
        btn3_ph.button('👥 Team', key='team_top')
    elif page.startswith('Year'):
        right_container = btn3_ph.container()
        b1, b2 = right_container.columns([1, 1])
        b1.button('🏠 Home', key='home_top')
        b2.button('📊 Visualization', key='viz_top')
    elif page == 'Visualization':
        right_container = btn3_ph.container()
        b1, b2 = right_container.columns([1, 1])
        b1.button('🏠 Home', key='home_top_viz')
        b2.button('🔙 Back to Predictions', key='back_top')
    elif page == 'Team':
        btn3_ph.button('🏠 Home', key='home_top_team')


# --- Page bodies ---
if st.session_state.page == 'Home':
    st.markdown("### Welcome to the Retail Demand Forecasting Dashboard")
    st.write('This dashboard uses our production **Prophet** model loaded directly from the MinIO Data Lake (`gold/models/prophet/prophet_models.pkl`).')
    st.write('Please upload your daily sales data (CSV format: `date`, `store`, `item`...).')

    left, right = st.columns([3, 1])
    with left:
        uploaded_file = st.file_uploader('Upload CSV file', type=['csv'], key='uploaded_file')
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error('Error reading CSV: ' + str(e))
        else:
            st.session_state.uploaded_df = None

    if st.session_state.uploaded_df is not None and not st.session_state.uploaded_df.empty:
        st.subheader('Preview of uploaded data')
        st.dataframe(st.session_state.uploaded_df.head(100), height=250)
        
        if st.button('🚀 Run Prophet Forecast on Uploaded Data', type='primary'):
            with st.spinner("Loading Prophet models from MinIO and running inference..."):
                st.session_state.predictions_df = generate_predictions(st.session_state.uploaded_df)
                st.session_state.page = predictions_page_name()
                st.rerun()


elif st.session_state.page.startswith('Year'):
    df = st.session_state.predictions_df
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning('No predictions available. Go to Home and click Make Predictions.')
        if st.button('Back to Home', key='back_no_preds'):
            go_to('Home')
    else:
        for c in ['year', 'week', 'store', 'item']:
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors='ignore')
                except Exception:
                    pass

        st.sidebar.header('Filters')
        years = sorted(df['year'].unique().tolist()) if 'year' in df.columns else []
        sel_year = st.sidebar.selectbox('Year', years, index=0 if years else 0)
        stores = sorted(df['store'].unique().tolist()) if 'store' in df.columns else []
        sel_stores = st.sidebar.multiselect('Store', stores, default=stores)
        items = sorted(df['item'].unique().tolist()) if 'item' in df.columns else []
        sel_items = st.sidebar.multiselect('Item', items, default=items)
        week_min = int(df['week'].min()) if 'week' in df.columns else 1
        week_max = int(df['week'].max()) if 'week' in df.columns else 52
        sel_week_range = st.sidebar.slider('Week range', week_min, week_max, (week_min, week_max))

        filtered = filter_predictions(df, year=sel_year, stores=sel_stores, items=sel_items, week_range=sel_week_range)

        st.subheader('Weekly Sales Predictions (Prophet)')
        st.dataframe(filtered, height=600)
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download filtered predictions', data=csv, file_name=f'prophet_predictions_2017.csv', mime='text/csv')

        st.divider()
        st.subheader("🔍 Production Monitoring Check")
        st.write("Click below to run the MLOps monitoring suite. This downloads the actual ground-truth sales from the MinIO Gold bucket to calculate our real performance and test for statistical drift.")

        if st.button("🚨 Run Monitor Check", type="secondary"):
            with st.spinner("Initializing PySpark and connecting to MinIO Data Lake..."):
                try:
                    import sys
                    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    if proj_root not in sys.path:
                        sys.path.insert(0, proj_root)
                    
                    from config.spark_config import get_spark
                    from src.pipeline.monitor import check_performance_drift, check_missing_data, get_baseline_rmse
                    
                    # 1. Get Actuals via PySpark from Silver Bucket
                    spark = get_spark("StreamlitMonitor")
                    silver_spark = spark.read.parquet("s3a://ecommerce-lake/silver/weekly_sales/")
                    silver_df = silver_spark.select("year", "week_of_year", "store", "item", "weekly_sales").toPandas()
                    silver_df = silver_df.rename(columns={"week_of_year": "week"})
                    
                    # 2. Merge Predictions with Actuals
                    merged = pd.merge(filtered, silver_df, on=["year", "week", "store", "item"], how="inner")
                    
                    if merged.empty:
                        st.warning("Could not find matching actual sales in the Gold bucket for these predictions. Cannot calculate RMSE.")
                    else:
                        from sklearn.metrics import mean_squared_error
                        import numpy as np
                        
                        # 3. Run Checks
                        current_rmse = np.sqrt(mean_squared_error(merged["weekly_sales"], merged["sales_prediction"]))
                        is_perf_drift = check_performance_drift("Prophet", current_rmse)
                        
                        is_missing = check_missing_data(st.session_state.uploaded_df)
                        
                        # 4. Display Dashboard Metrics
                        col1, col3 = st.columns(2)
                        
                        baseline_rmse = get_baseline_rmse("Prophet")
                        if baseline_rmse is None:
                            baseline_rmse = 30.41
                        
                        with col1:
                            st.metric("Current RMSE", f"{current_rmse:.2f}", delta=f"{current_rmse - baseline_rmse:.2f} from baseline", delta_color="inverse")
                            if is_perf_drift:
                                st.error("**Performance Drift Detected!**\nThe model's error has degraded by >10% compared to its training baseline. It is time to retrain.")
                            else:
                                st.success("**Performance OK**\nThe model is operating within acceptable error bounds.")
                                
                        with col3:
                            st.metric("Missing Data", "Failed" if is_missing else "Passed")
                            if is_missing:
                                st.error("**Missing Data Detected!**\nMore than 1% of the newly uploaded rows contain null values. Check the upstream pipeline.")
                            else:
                                st.success("**Data Quality OK**\nMissing values are within the acceptable <1% threshold.")
                        
                        st.info("💡 **Interpretation:** The **RMSE** tracks our Prophet model's exact accuracy. If it exceeds 10% degradation from the baseline stored in our training metrics, the model must be retrained. The **Missing Data** check ensures pipeline health before generating predictions.")
                        
                except Exception as e:
                    st.error(f"Error running monitor: {str(e)}")


elif st.session_state.page == 'Visualization':
    df = st.session_state.predictions_df
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning('No predictions to visualize. Please generate predictions first on the Home page.')
        if st.button('Back to Home', key='viz_back_home'):
            go_to('Home')
    else:
        st.sidebar.header('Filters')
        years = sorted(df['year'].unique().tolist()) if 'year' in df.columns else []
        sel_year = st.sidebar.selectbox('Year', years, index=0 if years else 0)
        stores = sorted(df['store'].unique().tolist()) if 'store' in df.columns else []
        sel_stores = st.sidebar.multiselect('Store', stores, default=stores)
        items = sorted(df['item'].unique().tolist()) if 'item' in df.columns else []
        sel_items = st.sidebar.multiselect('Item', items, default=items)
        week_min = int(df['week'].min()) if 'week' in df.columns else 1
        week_max = int(df['week'].max()) if 'week' in df.columns else 52
        sel_week_range = st.sidebar.slider('Week range', week_min, week_max, (week_min, week_max))

        filtered = filter_predictions(df, year=sel_year, stores=sel_stores, items=sel_items, week_range=sel_week_range)
        if filtered.empty:
            st.warning('No data after applying filters.')
        else:
            if 'week' in filtered.columns:
                try:
                    filtered['week'] = pd.to_numeric(filtered['week'], errors='coerce')
                except Exception:
                    pass

            weekly_store = filtered.groupby(['week', 'store'], as_index=False)['sales_prediction'].sum()
            fig_store = px.line(
                weekly_store,
                x='week',
                y='sales_prediction',
                color='store',
                title='Weekly Sales Prediction by Store (Prophet)',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            weekly_item = filtered.groupby(['week', 'item'], as_index=False)['sales_prediction'].sum()
            fig_item = px.line(
                weekly_item,
                x='week',
                y='sales_prediction',
                color='item',
                title='Weekly Sales Prediction by Item (Prophet)',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            item_totals = filtered.groupby('item', as_index=False)['sales_prediction'].sum().sort_values('sales_prediction', ascending=False)
            fig_item_bar = px.bar(
                item_totals,
                x='item',
                y='sales_prediction',
                title='Total Predicted Sales by Item',
                color='item',
                color_discrete_sequence=px.colors.sequential.Viridis,
            )

            store_totals = filtered.groupby('store', as_index=False)['sales_prediction'].sum().sort_values('sales_prediction', ascending=False)
            fig_store_bar = px.bar(
                store_totals,
                x='store',
                y='sales_prediction',
                title='Total Predicted Sales by Store',
                color='store',
                color_discrete_sequence=px.colors.sequential.Viridis,
            )

            l1, l2 = st.columns(2)
            l1.plotly_chart(fig_store, width='stretch')
            l2.plotly_chart(fig_item, width='stretch')

            b1, b2 = st.columns(2)
            b1.plotly_chart(fig_item_bar, width='stretch')
            b2.plotly_chart(fig_store_bar, width='stretch')


elif st.session_state.page == 'Team':
    team_data = [
        {'Role': 'Data Engineer (Ingestion & Storage)\n\nData Analyst (Exploration & Insights)', 'ID': '22010038','Name': 'احمد محمد محمود محمود محمدى'},
        {'Role': 'Machine Learning Engineer (Model Development)', 'ID': '22010056', 'Name': 'الحسين ياسر ابراهيم السيد'},
        {'Role': 'Big Data Engineer (Optimization & Performance)', 'ID': '22011562', 'Name': 'عمر حافظ مأمون محمد'},
        {'Role': 'MLOps Engineer (Deployment & Monitoring)', 'ID': '22010027', 'Name': 'أحمد عماد عبد الفتاح عبد الغني'},
    ]
    team_df = pd.DataFrame(team_data)
    st.table(team_df)

else:
    go_to('Home')
