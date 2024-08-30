# CSV Analyzer and Question Answering Chatbot

## Workflow

### User Interface Interaction
- **Access**: Users access the application through a web-based interface designed with Streamlit.
- **CSV Upload**: The interface includes an option to upload CSV files, which users can select and upload for analysis.

### Data Upload and Parsing
- **Parsing**: Upon uploading a CSV file, Pandas is used to parse the file into a DataFrame.
- **Column Detection**: The system automatically detects the types of columns (numeric, categorical, datetime) to tailor subsequent processing.

### Statistical Analysis
- **Statistics Computation**: Basic statistics such as mean, median, mode, and standard deviation are calculated for numeric columns.
- **UI Display**: These statistics are displayed in an expandable section on the user interface for easy access and review.

### Dynamic Insights Generation
- **Visualization**: Depending on the structure of the data, the application generates various types of visualizations:
  - Time series plots for numeric columns with a date column.
  - Bar charts, box plots, histograms, and correlation heatmaps as appropriate.
- **Tools Used**: Matplotlib and Plotly Express are employed to create these dynamic and interactive graphs.
- **User Interaction**: Users can interact with these visualizations through a dropdown menu that allows selection of different insights.

### Question Answering Chatbot
- **NLP API**: The Groq API, using the Mixtral-8x7b-32768 LLM model, powers the natural language processing capabilities of the chatbot.
- **User Queries**: Users can input questions about the data in natural language. The chatbot processes these queries, generates responses, and can also produce plots based on the queries.
- **Chat History**: The chat interface maintains a history of interactions, which helps provide context for new queries and allows users to reference previous interactions.

