import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from groq import Groq
import re

GROQ_API_KEY = "api key"
client = Groq(api_key=GROQ_API_KEY)

def generate_response(df, question):

    data_summary = df.describe().to_string()
    
   
    data_sample = df.head().to_string()
    
    main_prompt = f"""
    You are an AI assistant analyzing CSV data. The CSV has these columns: {', '.join(df.columns)}.

    Data summary:
    {data_summary}

    Data sample:
    {data_sample}

    User question: {question}

    Rules:
    1. If the user explicitly asks for a plot, graph, or visualization, ONLY provide a plot suggestion. Do not give any other information.
    2. If the user does not mention plot, graph, or visualize, provide a concise and relevant answer without any plot suggestions.
    3. For mathematical questions (e.g., total, average, sum, etc.), perform the calculation and provide the precise numerical answer, not the approach to solve it.
    4. When asked for a plot, suggest an appropriate plot type based on the data and question.
    5. IMPORTANT: When suggesting plots, ONLY use columns that exist in the dataset. The available columns are: {', '.join(df.columns)}. Do not suggest columns that are not in this list.

    Answer:
    """
    
    plot_instructions = """
    If a plot is requested, use the following format for your response:
    PLOT:
    Type: [plot type: line, bar, scatter, histogram, box, violin, heatmap, 3d scatter]
    Columns:
    - x: [column name]
    - y: [column name] (omit for histogram)
    - z: [column name] (only for 3d scatter)
    - color: [column name] (optional)
    - size: [column name] (optional)
    - hover_data: [column names] (optional)
    - facet_col: [column name] (optional)
    - facet_row: [column name] (optional)
    """

    is_plot_question = any(word in question.lower() for word in ['plot', 'graph', 'visualize', 'chart'])
    

    prompt = main_prompt + plot_instructions if is_plot_question else main_prompt
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=1000,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in generating response: {e}")
        return None

def generate_dynamic_insights(df):
    insights = {}
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    
    if len(date_cols) > 0:
        date_col = date_cols[0]
        for col in numeric_cols:
            insights[f'Total {col} Over Time'] = px.line(df, x=date_col, y=col, title=f'Total {col} Over Time')


    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            insights[f'Total {num_col} by {cat_col}'] = px.bar(df.groupby(cat_col)[num_col].sum().reset_index(), x=cat_col, y=num_col, title=f'Total {num_col} by {cat_col}')

    
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            insights[f'{num_col} Distribution by {cat_col}'] = px.box(df, x=cat_col, y=num_col, title=f'{num_col} Distribution by {cat_col}')


    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        insights["Correlation Heatmap"] = px.imshow(corr_matrix, title="Correlation Heatmap")

    return insights

def calculate_statistics(df):
    stats = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        stats[column] = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'mode': df[column].mode().iloc[0] if not df[column].mode().empty else np.nan,
            'std': df[column].std()
        }
    return stats

def parse_plot_info(response):
    plot_info = {'Type': '', 'Columns': {}}
    if 'PLOT:' in response:
        plot_section = response.split('PLOT:')[1].strip()
        type_match = re.search(r'Type:\s*(.+)', plot_section)
        if type_match:
            plot_info['Type'] = type_match.group(1).strip()
        
        column_section = plot_section.split('Columns:')[-1].strip()
        for line in column_section.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace('-', '').replace(' ', '')
                value = value.strip()
                if key in ['x', 'y', 'z', 'color', 'size', 'hover_data', 'facet_col', 'facet_row']:
                    if value.startswith('[') and value.endswith(']'):
                        value_list = [item.strip().strip("'\"") for item in value[1:-1].split(',')]
                        plot_info['Columns'][key] = value_list
                    else:
                        plot_info['Columns'][key] = value.strip("'\"")
    
    return plot_info

def create_interactive_plot(df, plot_info):
    plot_type = plot_info['Type'].lower().replace(' ', '')  
    columns = plot_info['Columns']
    
    try:
        plot_params = {'data_frame': df}
        
   
        valid_columns = {}
        for key, value in columns.items():
            if isinstance(value, str):
                if value in df.columns:
                    valid_columns[key] = value
                else:
                    st.warning(f"Column '{value}' specified for '{key}' is not in the DataFrame and will be ignored.")
            elif isinstance(value, list):
                valid_cols = [col for col in value if col in df.columns]
                if valid_cols:
                    valid_columns[key] = valid_cols
                else:
                    st.warning(f"None of the columns specified for '{key}' are in the DataFrame and will be ignored.")
        
        columns = valid_columns
        
        if plot_type == 'histogram':
            if 'x' not in columns:
                st.error("Histogram requires an 'x' column.")
                return None
            plot_params['x'] = columns['x']
        else:
            if 'x' not in columns or 'y' not in columns:
                st.error(f"Missing required columns for {plot_type} plot.")
                return None
            plot_params['x'] = columns['x']
            plot_params['y'] = columns['y']

        for key in ['color', 'hover_data', 'facet_col', 'facet_row', 'size']:
            if key in columns:
                plot_params[key] = columns[key]

        if plot_type in ['line', 'bar', 'barchart']:
            fig = px.bar(**plot_params) if plot_type in ['bar', 'barchart'] else px.line(**plot_params)
        elif plot_type == 'scatter':
            fig = px.scatter(**plot_params)
        elif plot_type == 'histogram':
            fig = px.histogram(**plot_params)
        elif plot_type == 'box':
            fig = px.box(**plot_params)
        elif plot_type == 'violin':
            fig = px.violin(**plot_params)
        elif plot_type == 'heatmap':
            if 'color' not in columns:
                columns['color'] = columns['y']
            fig = px.imshow(df.pivot(columns['x'], columns['y'], columns['color']))
        elif plot_type in ['3dscatter', '3dscatterplot']:
            if 'z' not in columns:
                st.error("3D scatter plot requires 'x', 'y', and 'z' columns.")
                return None
            fig = px.scatter_3d(**plot_params)
        else:
            st.error(f"Unsupported plot type: {plot_type}")
            return None

        fig.update_layout(title=f"{plot_type.capitalize()} plot")
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None


st.set_page_config(layout="wide", page_title="CSV Analyzer and Chatbot")

st.title("CSV Analyzer and Question Answering Chatbot")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Data Analysis")

       
        with st.expander("Dataset Preview", expanded=True):
            st.write(df.head())

        with st.expander("Statistical Analysis", expanded=True):
            stats = calculate_statistics(df)
            for column, stat in stats.items():
                st.subheader(f"Statistics for {column}")
                st.write(f"Mean: {stat['mean']:.2f}")
                st.write(f"Median: {stat['median']:.2f}")
                st.write(f"Mode: {stat['mode']:.2f}")
                st.write(f"Standard Deviation: {stat['std']:.2f}")
                st.write("---")

        
        st.subheader("Dynamic Insights")
        insights = generate_dynamic_insights(df)
        selected_insight = st.selectbox("Select an insight", list(insights.keys()))
        st.plotly_chart(insights[selected_insight], use_container_width=True)

    with col2:
        st.header("Question Answering Chatbot")
        st.write("Ask questions about the data:")

   
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.text_area("Q:", value=q, height=50, disabled=True, key=f"q_{i}")
            st.text_area("A:", value=a, height=100, disabled=True, key=f"a_{i}")
            st.write("---")

        user_question = st.text_input("Enter your question:")

        if user_question:
            response = generate_response(df, user_question)
            if response:
                if 'PLOT:' in response:
                    plot_info = parse_plot_info(response)
                    if plot_info['Type'] and plot_info['Columns']:
                        st.write("Generating plot based on the suggestion...")
                        fig = create_interactive_plot(df, plot_info)
                        if fig:
                            st.plotly_chart(fig)
                        else:
                            st.error("Could not generate the plot. The suggested columns might not exist in the dataset.")
                            st.write("Here's the original response from the AI:")
                            st.write(response)
                    else:
                        st.error("The AI suggested a plot but didn't provide enough information to create it.")
                        st.write("Here's the original response from the AI:")
                        st.write(response)
                else:
                    st.write("Answer:", response)
                
                st.session_state.chat_history.append((user_question, response))

    st.sidebar.header("Download Data")
    if st.sidebar.button("Download Analyzed Data"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv",
        )