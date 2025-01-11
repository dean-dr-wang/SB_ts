import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import timedelta
from time import sleep

# Set page configuration
st.set_page_config(
    page_title="Vessel Time Series Analysis",
    layout="wide",
    page_icon="📊"
)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("global_vessel_aggragated.csv (transform applied).csv", parse_dates=["date"])
    return data

data = load_data()

# Sidebar Filters
st.sidebar.title("🔧 Filters")
laden_status = st.sidebar.multiselect("Select Laden Status", options=data['laden_status'].unique(), default=data['laden_status'].unique())
sub_segment = st.sidebar.multiselect("Select Sub Segment", options=data['sub_segment'].unique(), default=data['sub_segment'].unique())
zone_name = st.sidebar.multiselect("Select Zone Name", options=data['zone_name'].unique(), default=data['zone_name'].unique())
aggregation = st.sidebar.selectbox("Aggregation Level", options=["Daily", "Weekly", "Monthly", "Quarterly"], index=0)
start_date, end_date = st.sidebar.date_input("Select Date Range", [data['date'].min(), data['date'].max()], label_visibility="collapsed")

# Filter data
data_filtered = data[
    (data['laden_status'].isin(laden_status)) &
    (data['sub_segment'].isin(sub_segment)) &
    (data['zone_name'].isin(zone_name)) &
    (data['date'] >= pd.Timestamp(start_date)) &
    (data['date'] <= pd.Timestamp(end_date))
]

# Aggregate data based on the selected frequency
if not data_filtered.empty:
    if aggregation == "Weekly":
        data_aggregated = data_filtered.resample('W-MON', on='date').sum().reset_index()
        forecast_periods = 13  # 13 weeks for a quarter
        freq = 'W'
    elif aggregation == "Monthly":
        data_aggregated = data_filtered.resample('M', on='date').sum().reset_index()
        forecast_periods = 3  # 3 months for a quarter
        freq = 'M'
    elif aggregation == "Quarterly":
        data_aggregated = data_filtered.resample('Q', on='date').sum().reset_index()
        forecast_periods = 1  # 1 quarter
        freq = 'Q'
    else:  # Daily
        data_aggregated = data_filtered
        forecast_periods = 90  # ~90 days for a quarter
        freq = 'D'
else:
    data_aggregated = pd.DataFrame()

# Color mapping for laden status
color_map = {
    'Laden': '#FF5733',  # Orange
    'Ballast': '#33FF57',  # Green
    'Unknown': '#3357FF',  # Blue
    'Part Laden': '#FF33A1',  # Pink
    'Other': '#A133FF'  # Purple
}

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🗺️ Zone Map", "🤖 Chatbot"])

with tab1:
    st.subheader(f"Vessel Count Over Time by Laden Status ({aggregation}) with Forecast")
    if not data_aggregated.empty:
        # First aggregate by date and laden_status
        if aggregation == "Weekly":
            data_grouped = data_filtered.groupby([pd.Grouper(key='date', freq='W-MON'), 'laden_status'])['vessel_count_sum'].sum().reset_index()
            freq = 'W'
            periods = 13  # 13 weeks for a quarter
        elif aggregation == "Monthly":
            data_grouped = data_filtered.groupby([pd.Grouper(key='date', freq='M'), 'laden_status'])['vessel_count_sum'].sum().reset_index()
            freq = 'M'
            periods = 3  # 3 months for a quarter
        elif aggregation == "Quarterly":
            data_grouped = data_filtered.groupby([pd.Grouper(key='date', freq='Q'), 'laden_status'])['vessel_count_sum'].sum().reset_index()
            freq = 'Q'
            periods = 1  # 1 quarter
        else:  # Daily
            data_grouped = data_filtered.groupby(['date', 'laden_status'])['vessel_count_sum'].sum().reset_index()
            freq = 'D'
            periods = 90  # ~90 days for a quarter

        # Create figure for combined actual and forecast
        fig = go.Figure()

        # Get unique laden statuses
        laden_statuses = data_grouped['laden_status'].unique()
        
        # Color mapping for consistency
        colors = px.colors.qualitative.Set1[:len(laden_statuses)]
        color_map = dict(zip(laden_statuses, colors))

        # For each laden status, create forecast and add to plot
        for laden_status in laden_statuses:
            # Filter data for current laden status
            status_data = data_grouped[data_grouped['laden_status'] == laden_status].copy()
            
            # Prepare data for Prophet
            prophet_data = status_data.rename(columns={'date': 'ds', 'vessel_count_sum': 'y'})
            
            # Create and fit Prophet model
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            m.fit(prophet_data)

            # Create future dates for prediction
            future = m.make_future_dataframe(periods=periods, freq=freq)
            forecast = m.predict(future)

            # Add actual values to plot
            fig.add_trace(go.Scatter(
                x=status_data['date'],
                y=status_data['vessel_count_sum'],
                name=f"{laden_status} (Actual)",
                mode='lines',
                line=dict(color=color_map[laden_status])
            ))

            # Add forecasted values
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tail(periods),
                y=forecast['yhat'].tail(periods),
                name=f"{laden_status} (Forecast)",
                mode='lines',
                line=dict(color=color_map[laden_status], dash='dash')
            ))

            # Add confidence intervals
            fill_color = color_map.get(laden_status, 'rgb(0,0,0)')  # Default to black in rgb format
            if fill_color.startswith('rgb'):  # Handle rgb() format
                # Extract RGB values from the rgb() string
                r, g, b = map(int, fill_color[4:-1].split(','))
            else:  # Handle hex format (fallback if color_map contains hex)
                fill_color = fill_color.lstrip('#')
                r, g, b = tuple(int(fill_color[i:i+2], 16) for i in (0, 2, 4))

            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'].tail(periods), forecast['ds'].tail(periods)[::-1]]),
                y=pd.concat([forecast['yhat_upper'].tail(periods), forecast['yhat_lower'].tail(periods)[::-1]]),
                fill='toself',
                fillcolor=f"rgba({r}, {g}, {b}, 0.2)",  # Use RGB with alpha transparency
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{laden_status} (95% Confidence)",
                showlegend=True
            ))


        # Update layout
        fig.update_layout(
            title=f"Vessel Count Over Time with {periods} period Forecast ({aggregation})",
            xaxis_title="Date",
            yaxis_title="Vessel Count",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for the selected filters.")

with tab2:
    st.subheader("Vessel Count by Zone")
    if not data_filtered.empty:
        zone_summary = data_filtered.groupby('zone_name')['vessel_count_sum'].sum().reset_index()

        # Generate sub-zones for Gulf of Mexico
        sub_zones = []
        for _, row in zone_summary.iterrows():
            if "Gulf of Mexico" in row['zone_name']:
                for i in range(3):  # Generate 3 sub-zones per Gulf of Mexico zone
                    sub_zones.append({
                        'zone_name': f"{row['zone_name']} Sub-Zone {i + 1}",
                        'vessel_count_sum': row['vessel_count_sum'] / 3,
                        'latitude': np.random.uniform(23.0, 30.0),  # Latitude range for Gulf of Mexico
                        'longitude': np.random.uniform(-98.0, -82.0),  # Longitude range for Gulf of Mexico
                        'size': np.random.uniform(10, 50)  # Random sizes for visualization
                    })
            else:
                sub_zones.append({
                    'zone_name': row['zone_name'],
                    'vessel_count_sum': row['vessel_count_sum'],
                    'latitude': 0.0,
                    'longitude': 0.0,
                    'size': 10  # Default size
                })

        expanded_zones = pd.DataFrame(sub_zones)

        # Create the map visualization
        fig = px.scatter_mapbox(
            expanded_zones,
            lat='latitude',
            lon='longitude',
            size='size',
            color='vessel_count_sum',
            color_continuous_scale="Viridis",
            hover_name='zone_name',
            title="Vessel Count by Zone",
            mapbox_style="carto-positron",
            zoom=4
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with tab3:
    st.subheader("🤖 Chatbot")
    st.write("Ask the chatbot about the trends and market sentiment!")
    
    default_prompt = "Explain the trend in vessel counts and how it relates to market sentiment."
    user_input = st.text_input("Enter your question:", placeholder=default_prompt)
    
    if user_input:
        # Simulate a typing effect for the response
        response = """
        The vessel count trend indicates a steady increase over the last quarter, particularly for laden vessels. 
        This aligns with market insights suggesting a rise in commodity exports, driven by demand from Asia and Europe. 
        Additionally, the Gulf of Mexico region has shown notable activity, indicating strong oil and gas shipments.
        
        Sources: Lloyd's List, Trade Winds News, Bloomberg Shipping Reports.
        """
        
        st.write("### Chatbot Response:")
        
        # Display the response character by character
        response_placeholder = st.empty()  # Placeholder for typing effect
        typed_text = ""
        for char in response:
            typed_text += char
            response_placeholder.markdown(f"{typed_text}")
            sleep(0.03)  # Adjust speed of typing effect