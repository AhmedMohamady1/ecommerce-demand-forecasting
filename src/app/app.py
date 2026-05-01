import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ecommerce Demand Forecasting App", layout="wide")


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
    # Try the preferred API if available; otherwise continue and the UI
    # will render the new page in the same run.
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



# Safe query-param helpers (some Streamlit versions don't expose these experimental APIs)
def get_query_params():
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}


def set_query_params(**kwargs):
    try:
        st.experimental_set_query_params(**kwargs)
    except Exception:
        return


# Handle query params (used by the wide green Make Predictions link when supported)
params = get_query_params()
if params and 'make_predictions' in params:
    # require uploaded data to exist
    if st.session_state.get('uploaded_file') is None and st.session_state.get('uploaded_df') is None:
        st.warning('Please upload a CSV file before making predictions.')
        set_query_params()
    else:
        # Actual model inference should populate `predictions_df` here.
        # For now create an empty predictions dataframe with expected columns.
        st.session_state.predictions_df = pd.DataFrame(
            columns=["year", "week", "store", "item", "sales_prediction"]
        )
        go_to(predictions_page_name())


# CSS for wide green Make Predictions link
st.markdown(
    """
    <style>
     a.make-preds{display:inline-block;text-align:center;background:#28a745;color:white;padding:14px 20px;border-radius:6px;text-decoration:none;font-weight:600;width:100%;}
     a.make-preds:hover{opacity:0.9;}
     /* tighten top spacing so header appears nearer the top */
     div.block-container{padding-top:0.5rem;}
     div.block-container h1{margin-top:0.25rem;}
     /* prevent button text wrapping and ensure horizontal spacing */
.stButton>button {
    width: auto;             
    min-width: 100px;   
    white-space: nowrap;
    padding: 6px 20px;  
    font-weight: 600;
    margin-left: 6px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
}</style>
    """,
    unsafe_allow_html=True,
)


# --- Top header + page-specific right-aligned buttons ---
col1, col2, col3 = st.columns([7, 1, 2.5], vertical_alignment="bottom")
title_ph = col1.empty()
btn2_ph = col2.empty()
btn3_ph = col3.empty()

# current page
page = st.session_state.get('page', 'Home')

# Render buttons into placeholders based on current page and capture clicks
home_top_clicked = False
team_top_clicked = False
viz_top_clicked = False
back_top_clicked = False
home_top_viz_clicked = False
home_top_team_clicked = False

if page == 'Home':
    team_top_clicked = btn3_ph.button('👥 Team', key='team_top')
elif page.startswith('Year'):
    # group the two action buttons into the right container so they stay together
    right_container = btn3_ph.container()
    b1, b2 = right_container.columns([1, 1])
    home_top_clicked = b1.button('🏠 Home', key='home_top')
    viz_top_clicked = b2.button('📊 Visualization', key='viz_top')
elif page == 'Visualization':
    # group Home and Back in the right container (Home left, Back right)
    right_container = btn3_ph.container()
    b1, b2 = right_container.columns([1, 1])
    home_top_viz_clicked = b1.button('🏠 Home', key='home_top_viz')
    back_top_clicked = b2.button('🔙 Back to Predictions', key='back_top')
elif page == 'Team':
    # right-align the Home button on the Team page
    home_top_team_clicked = btn3_ph.button('🏠 Home', key='home_top_team')

# Persistent title
title_ph.title('Ecommerce Demand Forecasting App')

# Determine new page if any button was clicked
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
    # update session state so the rest of the script renders the selected page
    st.session_state.page = new_page
    page = new_page
    # clear placeholders and re-render them for the new page state so header updates immediately
    title_ph.empty()
    btn2_ph.empty()
    btn3_ph.empty()
    # reuse the same placeholders to render the updated header/buttons
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
    st.write('Upload weekly sales data CSV')

    left, right = st.columns([3, 1])
    with left:
        uploaded_file = st.file_uploader('Upload CSV file (columns: year,week,store,item,...)', type=['csv'], key='uploaded_file')
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error('Error reading CSV: ' + str(e))
        else:
            st.session_state.uploaded_df = None

    # show preview and wide Make Predictions only when data is present
    if st.session_state.uploaded_df is not None and not st.session_state.uploaded_df.empty:
        st.subheader('Preview of uploaded data')
        st.dataframe(st.session_state.uploaded_df.head(200), height=300)
        st.markdown('<br/>', unsafe_allow_html=True)
        # If the Streamlit build supports query params, render a wide anchor that sets them.
        # Otherwise show a regular Streamlit button as a fallback.
        if hasattr(st, 'experimental_get_query_params'):
            st.markdown('<a href="?make_predictions=1" class="make-preds">🚀 Make Predictions</a>', unsafe_allow_html=True)
        else:
            if st.button('🚀 Make Predictions', key='make_preds_btn'):
                # Trigger prediction flow — replace with model inference integration.
                st.session_state.predictions_df = pd.DataFrame(
                    columns=["year", "week", "store", "item", "sales_prediction"]
                )
                st.session_state.page = predictions_page_name()
                st.rerun()


elif st.session_state.page.startswith('Year'):
    df = st.session_state.predictions_df
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning('No predictions available. Go to Home and upload data, then click Make Predictions.')
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

        st.subheader('Predictions table')
        st.dataframe(filtered, height=600)
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button('Download filtered predictions', data=csv, file_name=f'predictions.csv', mime='text/csv')


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
            # ensure week is numeric for proper ordering on x-axis
            if 'week' in filtered.columns:
                try:
                    filtered['week'] = pd.to_numeric(filtered['week'], errors='coerce')
                except Exception:
                    pass

            # Weekly sales per store (one line per store)
            weekly_store = filtered.groupby(['week', 'store'], as_index=False)['sales_prediction'].sum()
            fig_store = px.line(
                weekly_store,
                x='week',
                y='sales_prediction',
                color='store',
                title='Weekly Sales Prediction by Store',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            # Weekly sales per item (one line per item)
            weekly_item = filtered.groupby(['week', 'item'], as_index=False)['sales_prediction'].sum()
            fig_item = px.line(
                weekly_item,
                x='week',
                y='sales_prediction',
                color='item',
                title='Weekly Sales Prediction by Item',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            # Bar charts (keep existing but add colorful palettes)
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

            # Layout: two line plots on top, two bar charts below
            l1, l2 = st.columns(2)
            l1.plotly_chart(fig_store, width='stretch')
            l2.plotly_chart(fig_item, width='stretch')

            b1, b2 = st.columns(2)
            b1.plotly_chart(fig_item_bar, width='stretch')
            b2.plotly_chart(fig_store_bar, width='stretch')


elif st.session_state.page == 'Team':
    # --- EDIT THE `team_data` LIST BELOW TO FILL YOUR TEAM DETAILS IN THE CODE ---
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
