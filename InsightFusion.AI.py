import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
from autogluon.tabular import TabularPredictor
import seaborn as sns
import matplotlib.pyplot as plt
import pygwalker as pyg
load_dotenv(".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
import plotly.io as pio
pio.templates.default = "plotly_dark"
import base64
background_image = Image.open("background wallpaper\\purple&black.jpeg")
def get_image_base64(image):
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()
background_image_base64 = get_image_base64(background_image)
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}
    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .logo {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }}
    .logo img {{
        width: 150px;
        margin-right: 20px;
    }}
    .author-section {{
        margin-top: 15px;
    }}
    .author-section a {{
        margin-right: 40px;
    }}
    .author-section img {{
        width: 50px;
        height: 50px;
    }}
    .summary {{
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    .funding {{
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    .license {{
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    .sidebar-button {{
        display: block;
        width: 100%;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
        background-color: #0677a1;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        text-decoration: none;
        font-size: 14px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }}
    .sidebar-button:hover {{
        background-color: #055a7a;
        transform: scale(1.02);
    }}
    .sidebar-separator {{
        margin: 20px 0;
        border-top: 2px solid #0677a1;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Data Transformation", "Make Predictions", "Generative AI", "DashBoard"])
st.sidebar.markdown('<hr class="sidebar-separator">', unsafe_allow_html=True)
st.sidebar.title("Learning & Documentation")
st.sidebar.markdown("### Explore Resources:")
st.sidebar.markdown(
    """
    <a href="https://pandas.pydata.org/pandas-docs/stable/" target="_blank" class="sidebar-button">Pandas Documentation</a>
    <a href="https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114" target="_blank" class="sidebar-button">Feature Engineering</a>
    <a href="https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-3e5f0f276b1e" target="_blank" class="sidebar-button">Data Cleaning</a>
    <a href="https://ai.google.dev/" target="_blank" class="sidebar-button">Generative AI (Gemini)</a>
    <a href="https://auto.gluon.ai/stable/index.html" target="_blank" class="sidebar-button">AutoGluon Documentation</a>
    <a href="https://docs.kanaries.net/pygwalker" target="_blank" class="sidebar-button">PyGWalker Documentation</a>
    """,
    unsafe_allow_html=True,
)

def home_page():
    col1, col2 = st.columns([1, 4]) 
    with col1:
        st.image(Image.open("logo.png"), width=200)  
    with col2:
        st.markdown(
            """
            <div>
                <h2>InsightFusion.AI: Enhanced Generative AI Data Pipeline</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### App Purpose")
    st.markdown("""
    This app leverages Generative AI to streamline data preprocessing and predictive analytics, transforming raw data into actionable insights with ease. It offers a comprehensive suite of tools for data transformation (e.g., renaming and dropping columns, imputing missing values), exploratory data analysis (EDA) with interactive visualizations, and predictive modeling using our advanced machine learning models. With an integrated, interactive dashboard, users can seamlessly analyze data, gain insights, and accelerate data-driven decision-making.
    """)
    st.markdown("---")

    st.markdown("### Workflow")
    st.image(Image.open("Workflow_image.png"), use_container_width=True)  # Updated parameter
    st.markdown("---")


    st.markdown("### Author")
    st.markdown("Please feel free to contact us with any issues, comments, or questions.")
    st.markdown(
        """
        <div class="author-section">
            <a href="https://twitter.com/YourTwitterHandle" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/twitter.png" alt="Twitter">
            </a>
            <a href="https://github.com/SreeHariVaithiyanathan" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/github.png" alt="GitHub">
            </a>
            <a href="https://www.linkedin.com/in/sree-hari-vaithiyanathan/" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
            </a>
            <a href="mailto:sreehari052004@gmail.com" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/gmail.png" alt="Email">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
def data_transformation_page():
    st.title("Data Transformation")
    st.markdown("### Preprocess your data here.")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            
            data = pd.read_csv(uploaded_file, encoding='latin-1')
            st.session_state["data"] = data  
            
            
            st.subheader("Initial Analysis :chart:")

            
            st.write(" ### :information_source: **Data Info :** ")
            with st.expander("Show Data Info"):
                st.write(f"Shape of the dataset: {data.shape}")

                
                col1, col2 = st.columns(2)

                
                with col1:
                    st.write("Columns and their data types:")
                    st.write(data.dtypes)

                
                with col2:
                    st.write("Number of non-null values in each column:")
                    st.write(data.notnull().sum())

                st.write("Summary statistics: :1234:")
                st.write(data.describe())

            
            st.subheader("Uploaded Dataset :wrench:")
            st.dataframe(data)

            
            st.sidebar.subheader("Preprocessing Options :gear:")
            st.sidebar.markdown("---")  
            
           
            if st.sidebar.checkbox("Rename Columns"):
                renamed_columns = {}
                for col in data.columns:
                    new_name = st.sidebar.text_input(f"New name for {col}:", col)
                    renamed_columns[col] = new_name
                data.rename(columns=renamed_columns, inplace=True)
            
            
            if st.sidebar.checkbox("Drop Columns"):
                cols_to_drop = st.sidebar.multiselect("Select Columns to Drop", data.columns)
                data.drop(cols_to_drop, axis=1, inplace=True)
            
           
            if st.sidebar.checkbox("Remove Special Characters"):
                st.sidebar.subheader("Select Features and Special Characters to Remove")
                special_char_features = st.sidebar.multiselect("Select Features to Remove Special Characters From", data.columns)
                special_chars = st.sidebar.text_input("Enter Special Characters to Remove (e.g., @$%&*!)")
                if special_char_features and special_chars:
                    special_chars_regex = f"[{re.escape(special_chars)}]"
                    for col in special_char_features:
                        data[col] = data[col].apply(lambda x: re.sub(special_chars_regex, '', str(x)) if pd.notnull(x) else x)
            
        
            if st.sidebar.checkbox("Convert Data Types"):
                st.sidebar.subheader("Select Features and Data Types for Conversion")
                conversion_options = ["int64", "float64", "datetime64[ns]", "object"]
                for col in data.columns:
                    new_type = st.sidebar.selectbox(f"Convert {col} to:", conversion_options, index=conversion_options.index(str(data[col].dtype)))
                    if new_type == "datetime64[ns]":
                        try:
                            data[col] = pd.to_datetime(data[col], errors='coerce')
                        except ValueError:
                            st.warning(f"Failed to convert {col} to datetime. It may contain invalid values.")
                    else:
                        data[col] = data[col].astype(new_type)
            
            
            numeric_cols = data.select_dtypes(include=np.number).columns
            categorical_cols = data.select_dtypes(exclude=np.number).columns
            
            
            if st.sidebar.checkbox("Impute Missing Values"):
                imputation_strategy_numeric = st.sidebar.radio("Select Imputation Strategy for Numeric Features", ["mean", "median"])
                imputation_strategy_categorical = st.sidebar.radio("Select Imputation Strategy for Categorical Features", ["most_frequent"])
                features_to_impute_numeric = st.sidebar.multiselect("Select Numeric Features to Impute", numeric_cols)
                features_to_impute_categorical = st.sidebar.multiselect("Select Categorical Features to Impute", categorical_cols)

                if features_to_impute_numeric:
                    for col in features_to_impute_numeric:
                        if imputation_strategy_numeric == "mean":
                            data[col].fillna(data[col].mean(), inplace=True)
                        elif imputation_strategy_numeric == "median":
                            data[col].fillna(data[col].median(), inplace=True)

                if features_to_impute_categorical:
                    for col in features_to_impute_categorical:
                        if imputation_strategy_categorical == "most_frequent":
                            data[col].fillna(data[col].mode()[0], inplace=True)
            
           
            if st.sidebar.checkbox("Scale Data"):
                st.sidebar.subheader("Select Features to Scale")
                features_to_scale = st.sidebar.multiselect("Select Features to Scale", numeric_cols)
                scaler = StandardScaler()
                data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
            
            
            if st.sidebar.checkbox("Encode Categorical Values"):
                label_encoders = {}
                for col in categorical_cols:
                    label_encoders[col] = LabelEncoder()
                    data[col] = label_encoders[col].fit_transform(data[col])
                
                st.session_state["label_encoders"] = label_encoders
            
            
            st.subheader("Final Preprocessed Dataset :heavy_check_mark:")
            st.dataframe(data)

            
            st.subheader("Final Preprocessed Data Info")
            with st.expander("Show Final Preprocessed Data Info"):
                st.write(f"Shape of the final preprocessed dataset: {data.shape}")

                
                col1, col2 = st.columns(2)

               
                with col1:
                    st.write("Columns and their data types:")
                    st.write(data.dtypes)

              
                with col2:
                    st.write("Number of non-null values in each column:")
                    st.write(data.notnull().sum())

                st.write("Summary statistics:")
                st.write(data.describe())

        except Exception as e:
            st.error("An error occurred while reading the file: {}".format(str(e)))
    else:
        st.warning("Please upload a dataset to get started.")


def prediction_page():
    st.title("AutoGluon Predictions")
    st.markdown("### Make predictions using AutoGluon.")

    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]

        
        target_variable = st.selectbox("Select Target Variable", data.columns)

        
        problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

        if st.button("Run AutoGluon"):
            st.write("Running AutoGluon...")

           
            predictor = TabularPredictor(label=target_variable).fit(train_data=data)

           
            st.session_state["predictor"] = predictor

           
            st.write("### AutoGluon Leaderboard")
            leaderboard = predictor.leaderboard(data)
            st.write(leaderboard)

           
            st.session_state["leaderboard"] = leaderboard

      
        if "predictor" in st.session_state:
            predictor = st.session_state["predictor"]
            leaderboard = st.session_state["leaderboard"]

            
            model_names = leaderboard["model"]
            selected_model = st.selectbox("Select the Best Model from the Leaderboard", model_names)

            
            st.subheader("Make Predictions with Selected Model")
            st.write("Enter the feature values for prediction:")

            
            st.write("### Select Key Features for Prediction")
            key_features = st.multiselect("Select Key Features", data.columns.drop(target_variable))

            
            input_data = {}
            for feature in key_features:
                if data[feature].dtype == "object":
                    
                    unique_values = data[feature].unique()
                    input_data[feature] = st.selectbox(f"Select value for {feature}", unique_values)
                else:
                   
                    default_value = float(data[feature].median())  
                    input_data[feature] = st.text_input(f"Enter value for {feature}", value=str(default_value))

           
            for feature in data.columns:
                if feature != target_variable and feature not in key_features:
                    if data[feature].dtype == "object":
                      
                        input_data[feature] = data[feature].mode()[0]
                    else:
                        
                        input_data[feature] = data[feature].median()

            if st.button("Predict"):
                try:
                    
                    input_df = pd.DataFrame([input_data])

                    
                    for col in input_df.columns:
                        if col in st.session_state.get("label_encoders", {}):
                            
                            input_df[col] = st.session_state["label_encoders"][col].transform([input_df[col].iloc[0]])[0]
                        else:
                            input_df[col] = input_df[col].astype(data[col].dtype)

                    
                    missing_features = set(data.columns) - set(input_df.columns)
                    for feature in missing_features:
                        if feature != target_variable:
                            if data[feature].dtype == "object":
                                input_df[feature] = data[feature].mode()[0] 
                            else:
                                input_df[feature] = data[feature].median() 

                    
                    input_df = input_df[data.columns.drop(target_variable)]

                    
                    prediction = predictor.predict(input_df, model=selected_model)
                    st.write(f"### Prediction: {prediction[0]}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.warning("Please upload a dataset on the Data Transformation page first.")
import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv
from scipy.stats import iqr
load_dotenv(".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generative_ai_page():
    st.title("InsightFusion The Generative AI")
    st.markdown("#### Ask any question about your dataset and get real-time insights, visualizations, and analysis.")

    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="generative_ai_uploader")

    if uploaded_file is not None:
        try:
            
            data = pd.read_csv(uploaded_file, encoding='latin-1')
            st.session_state["data"] = data 

           
            st.subheader("Uploaded Dataset")
            st.dataframe(data.head(), use_container_width=True)

            
            user_query = st.text_area("Enter your question or request here (e.g., 'Perform EDA', 'What is the occupation in the dataset?', 'Show me a correlation heatmap'):")

            
            if st.button("Process Query"):
                if user_query:
                    try:
                        
                        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

                        
                        code_prompt = f"""
                       You are an expert Data Scientist and Python programmer. Your task is to generate executable Python code to analyze a given dataset based on the user query.

### Dataset Details:
- **Columns**: {data.columns.tolist()}
- **Statistical Summary**:
{data.describe()}

### User Query:
"{user_query}"

### **Your Code Generation Instructions:**
1. **Use the variable 'data'** directly for dataset operations. Do **not** load it from a file.
2. **Generate only valid, executable Python code**. Do **not** include explanations, comments, markdown, or unnecessary text.
3. **Ensure all required imports** (e.g., `from scipy.stats import iqr`, `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`).
4. **Handle non-numeric data gracefully**:
   - Skip non-numeric columns unless the operation requires them.
   - Convert non-numeric data to numeric (if applicable, e.g., encoding categorical data).
5. **Use appropriate Python libraries** based on the query:
   - `pandas` for data manipulation (`dataframe operations, aggregations`).
   - `numpy` for numerical computations.
   - `matplotlib` and `seaborn` for data visualization.
   - `scipy.stats` for statistical analysis.
6. **For visualizations, use Streamlit functions**:
   - `st.write()` for displaying results.
   - `st.pyplot()` for Matplotlib charts.
   - `st.plotly_chart()` for interactive Plotly visualizations.
7. **Optimize performance**:
   - Use vectorized operations (`df.apply()`, `df.groupby()`) instead of loops.
   - Avoid redundant computations.

### **Expected Output:**
The generated Python code should run without errors in a **Streamlit environment**, producing accurate outputs and visualizations relevant to the user query.

Now, generate the required Python code:
                        """
                        code_response = model.generate_content(code_prompt)
                        generated_code = code_response.text

                        generated_code = re.sub(r"```python|```", "", generated_code).strip()

                        
                        generated_code = re.sub(r"pd\.read_csv\(.*\)", "data", generated_code)

                        
                        generated_code = generated_code.replace("df", "data")

                        
                        st.subheader("Generated Python Code")
                        st.code(generated_code, language="python")

                       
                        st.subheader("Execution Results")
                        try:
                            
                            exec_globals = {
                                "pd": pd,
                                "np": np,
                                "plt": plt,
                                "sns": sns,
                                "px": px,
                                "data": data,
                                "st": st,
                                "iqr": iqr,  
                                "LabelEncoder": LabelEncoder,  
                                "StandardScaler": StandardScaler,  
                            }
                            exec_locals = {}

                            
                            exec(generated_code, exec_globals, exec_locals)

                            
                            if "plt" in exec_locals:
                                st.pyplot(exec_locals["plt"].gcf())
                            if "fig" in exec_locals:
                                st.plotly_chart(exec_locals["fig"], use_container_width=True) 
                            if "output" in exec_locals:
                                st.write(exec_locals["output"])
                        except Exception as e:
                            st.error(f"Error executing the generated code: {str(e)}")

                        
                        insights_prompt = f"""
                        Explain the following analysis or results in simple terms:
                        Analysis: {generated_code}
                        Dataset summary: {data.describe()}
                        """
                        insights_response = model.generate_content(insights_prompt)
                        st.subheader("Additional Insights")
                        st.write(insights_response.text)

                    except Exception as e:
                        st.error(f"An error occurred while processing your query: {str(e)}")
                else:
                    st.warning("Please enter a question or request to proceed.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {str(e)}")
    else:
        st.warning("Please upload a dataset to get started.")

load_dotenv()
os.environ["KANARIES_API_KEY"] = os.getenv("KANARIES_API_KEY")
def pygwalker_page():
    st.title("PyGWalker: Interactive Data Exploration")
    st.markdown("### Explore your data interactively using PyGWalker.")

    if "data" in st.session_state and st.session_state["data"] is not None:
        data = st.session_state["data"]

        
        st.subheader("Interactive Data Exploration")        
   
        if not os.getenv("KANARIES_API_KEY"):
            st.error("Kanaries API key is missing. Please add it to the `.env` file.")
        else:
            pyg_html = pyg.walk(data, env='Streamlit', return_html=True)
            st.components.v1.html(pyg_html, height=1000, scrolling=True)
       
        st.subheader("AI-Powered Insights")
        if st.button("Generate Insights with Gemini"):
            model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
            prompt = f"""
            Analyze the following dataset and provide key insights:
            Dataset columns: {data.columns.tolist()}
            Dataset summary: {data.describe()}
            """
            response = model.generate_content(prompt)
            st.write(response.text)
   
        st.subheader("Export Data")
        if st.button("Export Preprocessed Data as CSV"):
            data.to_csv("preprocessed_data.csv", index=False)
            st.success("Data exported successfully as `preprocessed_data.csv`.")
    else:
        st.warning("Please upload a dataset on the Data Transformation page first.")
if page == "Home":
    home_page()
elif page == "Data Transformation":
    data_transformation_page()
elif page == "Make Predictions":
    prediction_page()
elif page == "Generative AI":
    generative_ai_page()
elif page == "DashBoard":
    pygwalker_page()
