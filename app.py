import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import json
import numpy as np
import pandas as pd
import requests # Import the requests library

# Define colors for the candles
COLOR_BULL = 'rgba(38,166,154,0.9)'  # #26a69a - Green
COLOR_BEAR = 'rgba(239,83,80,0.9)'   # #ef5350 - Red

# Define colors for the bands
COLOR_SUPPORT_BAND = 'rgba(144, 238, 144, 0.2)' # Light Green with transparency
COLOR_RESISTANCE_BAND = 'rgba(255, 99, 71, 0.2)' # Tomato Red with transparency

# Corrected CSVFILE path: Assuming the CSV is in the same directory as app.py
# Added .csv extension
CSVFILE = 'TSLA_data - Sheet1 (1).csv'

# --- Function to load and preprocess data (for both chart and chatbot) ---
@st.cache_data # Cache data loading to improve performance
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, skiprows=0, parse_dates=['timestamp'], skip_blank_lines=True)
    df['time'] = df['timestamp'].view('int64') // 10**9
    df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)
    df.dropna(inplace=True)

    def parse_list_string(s):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    df['Support'] = df['Support'].apply(parse_list_string)
    df['Resistance'] = df['Resistance'].apply(parse_list_string)
    return df

# Load the full dataset once
full_df = load_and_preprocess_data(CSVFILE)

# --- Start of Streamlit App Layout ---
st.title("TSLA Stock Analysis Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Candlestick Chart", "Chatbot"])

with tab1:
    st.header("Candlestick Chart with Markers and Support/Resistance Bands")

    # Use only the first 330 rows for the chart
    df_chart = full_df.head(330).copy()

    # Export candlestick data to JSON format
    candles = json.loads(
        df_chart.filter(['open', 'high', 'low', 'close', 'time'], axis=1)
          .to_json(orient="records")
    )

    # --- Prepare Markers Data ---
    markers = []
    for index, row in df_chart.iterrows():
        marker_time = row['time']
        marker_direction = row['direction']

        if marker_direction == 'LONG':
            markers.append({
                "time": marker_time,
                "position": 'belowBar',
                "color": '#26a69a', # Green
                "shape": 'arrowUp',
                "text": 'LONG'
            })
        elif marker_direction == 'SHORT':
            markers.append({
                "time": marker_time,
                "position": 'aboveBar',
                "color": '#ef5350', # Red
                "shape": 'arrowDown',
                "text": 'SHORT'
            })
        elif pd.isna(marker_direction):
            markers.append({
                "time": marker_time,
                "position": 'inBar',
                "color": '#FFD700', # Gold/Yellow
                "shape": 'circle',
                "text": 'Neutral'
            })

    seriesCandlestickChart = [{
        "type": 'Candlestick',
        "data": candles,
        "options": {
            "upColor": '#26a69a',
            "downColor": '#ef5350',
            "borderVisible": False,
            "wickUpColor": '#26a69a',
            "wickDownColor": '#ef5350'
        },
        "markers": markers
    }]

    # --- Prepare Bands Data ---
    support_band_data = []
    resistance_band_data = []

    for index, row in df_chart.iterrows():
        band_time = row['time']

        if isinstance(row['Support'], list) and len(row['Support']) > 0:
            lower_support = min(row['Support'])
            upper_support = max(row['Support'])
            support_band_data.append({
                "time": band_time,
                "value": upper_support,
                "bottom": lower_support
            })

        if isinstance(row['Resistance'], list) and len(row['Resistance']) > 0:
            lower_resistance = min(row['Resistance'])
            upper_resistance = max(row['Resistance'])
            resistance_band_data.append({
                "time": band_time,
                "value": upper_resistance,
                "bottom": lower_resistance
            })

    support_band_series = {
        "type": 'Area',
        "data": support_band_data,
        "options": {
            "priceLineVisible": False,
            "lineVisible": False,
            "topColor": COLOR_SUPPORT_BAND,
            "bottomColor": COLOR_SUPPORT_BAND,
            "lastValueVisible": False,
            "crosshairMarkerVisible": False,
        }
    }

    resistance_band_series = {
        "type": 'Area',
        "data": resistance_band_data,
        "options": {
            "priceLineVisible": False,
            "lineVisible": False,
            "topColor": COLOR_RESISTANCE_BAND,
            "bottomColor": COLOR_RESISTANCE_BAND,
            "lastValueVisible": False,
            "crosshairMarkerVisible": False,
        }
    }

    chartOptions = {
        "layout": {
            "textColor": 'black',
            "background": {
                "type": 'solid',
                "color": 'white'
            }
        },
        "grid": {
            "vertLines": { "color": "rgba(197, 203, 206, 0.5)" },
            "horzLines": { "color": "rgba(197, 203, 206, 0.5)" }
        },
        "crosshair": {
            "mode": 0
        },
        "rightPriceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)"
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "timeVisible": True,
            "secondsVisible": False
        }
    }

    renderLightweightCharts([
        {
            "chart": chartOptions,
            "series": seriesCandlestickChart + [support_band_series, resistance_band_series]
        }
    ], 'candlestick_with_features')

with tab2:
    st.header("Chatbot Interface")
    st.write("Ask questions about the TSLA stock data!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Start of Data Analysis for specific questions ---
    # Longest consecutive 'LONG' period
    max_consecutive_long_days = 0
    current_consecutive_long_days = 0
    start_date_of_longest_period = None
    end_date_of_longest_period = None
    current_period_start_date = None

    full_df_sorted_for_analysis = full_df.sort_values(by='timestamp').reset_index(drop=True)

    for index, row in full_df_sorted_for_analysis.iterrows():
        if row['direction'] == 'LONG':
            current_consecutive_long_days += 1
            if current_period_start_date is None:
                current_period_start_date = row['timestamp']
        else:
            if current_consecutive_long_days > max_consecutive_long_days:
                max_consecutive_long_days = current_consecutive_long_days
                start_date_of_longest_period = current_period_start_date
                end_date_of_longest_period = full_df_sorted_for_analysis.loc[index - 1, 'timestamp'] if index > 0 else current_period_start_date
            current_consecutive_long_days = 0
            current_period_start_date = None

    if current_consecutive_long_days > max_consecutive_long_days:
        max_consecutive_long_days = current_consecutive_long_days
        start_date_of_longest_period = current_period_start_date
        end_date_of_longest_period = full_df_sorted_for_analysis.iloc[-1]['timestamp']

    formatted_start_date = start_date_of_longest_period.strftime('%Y-%m-%d') if start_date_of_longest_period is not None else "N/A"
    formatted_end_date = end_date_of_longest_period.strftime('%Y-%m-%d') if end_date_of_longest_period is not None else "N/A"

    longest_long_period_answer = f"The longest consecutive 'LONG' period was {max_consecutive_long_days} days, from {formatted_start_date} to {formatted_end_date}."

    # Calculate average 'open' price for days where 'direction' was 'SHORT' in 2024
    df_2024_short = full_df[(full_df['timestamp'].dt.year == 2024) & (full_df['direction'] == 'SHORT')]
    avg_open_short_2024 = df_2024_short['open'].mean()
    if not pd.isna(avg_open_short_2024):
        avg_open_short_2024_answer = f"The average 'open' price for days where the 'direction' was 'SHORT' in 2024 was ${avg_open_short_2024:.2f}."
    else:
        avg_open_short_2024_answer = "No 'SHORT' days found in 2024 to calculate the average 'open' price."

    # Find instances where 'close' price broke above 'Resistance' or below 'Support' bands
    breaches = []
    for index, row in full_df.iterrows():
        close_price = row['close']
        timestamp = row['timestamp'].strftime('%Y-%m-%d')

        # Check for breaches above Resistance
        if isinstance(row['Resistance'], list) and len(row['Resistance']) > 0:
            upper_resistance = max(row['Resistance'])
            if close_price > upper_resistance:
                breaches.append(f"On {timestamp}, the 'close' price (${close_price:.2f}) broke above the upper Resistance band (${upper_resistance:.2f}).")

        # Check for breaches below Support
        if isinstance(row['Support'], list) and len(row['Support']) > 0:
            lower_support = min(row['Support'])
            if close_price < lower_support:
                breaches.append(f"On {timestamp}, the 'close' price (${close_price:.2f}) broke below the lower Support band (${lower_support:.2f}).")

    if breaches:
        breaches_answer = "Here are the instances where the 'close' price broke above a 'Resistance' band or below a 'Support' band:\n" + "\n".join(breaches)
    else:
        breaches_answer = "No instances found where the 'close' price broke above a 'Resistance' band or below a 'Support' band."

    # --- End of Data Analysis for specific questions ---

    # Convert DataFrame to a string format suitable for the LLM
    data_summary = full_df[['timestamp', 'open', 'high', 'low', 'close', 'direction', 'Support', 'Resistance']].to_string()

    # Create a container for chat messages to keep them above the input box
    chat_messages_container = st.container()

    with chat_messages_container:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Place the chat input at the bottom of the app
    # This ensures it stays at the bottom regardless of chat history length
    if prompt := st.chat_input("Ask a question about the TSLA data...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Rerun to display the user's message immediately
        st.rerun()

    # Process the last message if it was from the user and not yet processed by assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Define a dictionary to map questions to pre-calculated answers
            pre_calculated_answers = {
                "longest consecutive period (in days) where the 'direction' was 'long'": longest_long_period_answer,
                "average 'open' price for days where 'direction' was 'short' in 2024": avg_open_short_2024_answer,
                "instances where the 'close' price broke above a 'resistance' band or below a 'support' band": breaches_answer
            }

            user_question = st.session_state.messages[-1]["content"].lower()

            # Check if the user's question matches a pre-calculated answer
            found_pre_calculated_answer = False
            for q_key, answer in pre_calculated_answers.items():
                if q_key in user_question:
                    full_response = answer
                    found_pre_calculated_answer = True
                    break

            if not found_pre_calculated_answer:
                # If not a pre-calculated question, send to LLM with general instructions
                llm_prompt = f"""
                You are a stock market data analyst. Answer questions about the provided TSLA stock data.
                The data is as follows:
                {data_summary}

                Based on the data provided, answer the following question:
                {st.session_state.messages[-1]["content"]}

                If you cannot directly answer the question based on the provided data or if it requires complex calculations you cannot perform, please state that you are unable to answer the question or that you are not capable of answering it. Do not attempt to explain how to calculate it or provide a generic response.
                """

                try:
                    api_key = "AIzaSyAJICDcGOM6q3Ud9Jkk4pSmSKSKWaeGbGA" # <--- REPLACE THIS WITH YOUR ACTUAL API KEY

                    if api_key == "AIzaSyAJICDcGOM6q3Ud9Jkk4pSmSKSKWaeGbGA" or not api_key:
                        full_response = "Please set your Gemini API key in the `api_key` variable within the `app.py` file to enable chatbot functionality."
                    else:
                        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

                        chat_history = []
                        chat_history.append({"role": "user", "parts": [{"text": llm_prompt}]})
                        payload = {"contents": chat_history}

                        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
                        response.raise_for_status()
                        result = response.json()

                        if result and result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                            full_response = result["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            full_response = "I couldn't get a response from the AI. Please try again."

                except requests.exceptions.RequestException as e:
                    full_response = f"An error occurred while communicating with the AI (requests error): {e}"
                except Exception as e:
                    full_response = f"An unexpected error occurred: {e}"

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
