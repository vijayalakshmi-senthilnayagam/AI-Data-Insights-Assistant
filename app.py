import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import tempfile
import os
import re
import json
from dotenv import load_dotenv


load_dotenv()
# ================================
# 🔹 Hugging Face Setup
# ================================
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"
client = InferenceClient(token=HF_TOKEN)

# ================================
# 🔹 Helper Functions
# ================================
def create_auto_chart(df, question, answer_text):
    """Automatically create charts based on data and question"""
    try:
        # Try to extract JSON suggestion first
        chart_json_match = re.search(r'\{[^{}]*"[^"]*"[^{}]*\}', answer_text)
        if chart_json_match:
            try:
                chart_info = json.loads(chart_json_match.group())
                chart_type = chart_info.get("chart_type", "").lower()
                x_col = chart_info.get("x")
                y_col = chart_info.get("y")
                
                if x_col in df.columns and y_col in df.columns:
                    return create_specific_chart(df, chart_type, x_col, y_col)
            except:
                pass
        
        # Fallback: Auto-detect based on data types and question
        return create_fallback_chart(df, question)
        
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

def create_specific_chart(df, chart_type, x_col, y_col):
    """Create specific chart type"""
    plt.figure(figsize=(10, 6))
    
    try:
        if chart_type == "bar":
            if len(df[x_col].unique()) > 15:
                # Aggregate for too many categories
                top_cats = df[x_col].value_counts().head(10).index
                df_filtered = df[df[x_col].isin(top_cats)]
                df_agg = df_filtered.groupby(x_col)[y_col].mean().sort_values(ascending=False)
            else:
                df_agg = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
            
            plt.bar(df_agg.index.astype(str), df_agg.values, color='skyblue', alpha=0.8)
            plt.xlabel(x_col)
            plt.ylabel(f'Average {y_col}')
            plt.title(f'Average {y_col} by {x_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        elif chart_type == "line":
            if df[x_col].dtype in ['object']:
                # Convert categorical to numeric for line plot
                df_agg = df.groupby(x_col)[y_col].mean().sort_index()
            else:
                df_agg = df.groupby(x_col)[y_col].mean().sort_index()
            
            plt.plot(df_agg.index.astype(str), df_agg.values, marker='o', linewidth=2, markersize=6)
            plt.xlabel(x_col)
            plt.ylabel(f'Average {y_col}')
            plt.title(f'{y_col} Trend by {x_col}')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        elif chart_type == "pie":
            counts = df[x_col].value_counts().head(8)  # Limit to top 8
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f'Distribution of {x_col}')
            
        elif chart_type == "scatter":
            plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} vs {x_col}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
        else:
            return create_fallback_chart(df, f"Show relationship between {x_col} and {y_col}")
        
        return plt
        
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {e}")
        return create_fallback_chart(df, f"Show {x_col} and {y_col}")

def create_fallback_chart(df, question):
    """Create automatic charts based on data analysis"""
    plt.figure(figsize=(10, 6))
    
    try:
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Analyze question keywords
        question_lower = question.lower()
        
        # Chart 1: If we have numeric columns, show distribution
        if numeric_cols:
            if any(word in question_lower for word in ['distribution', 'histogram', 'frequency']):
                col = numeric_cols[0]
                plt.hist(df[col].dropna(), bins=15, alpha=0.7, color='lightblue', edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {col}')
                
            # Chart 2: If we have both numeric and categorical, show bar chart
            elif categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                if len(df[cat_col].unique()) <= 15:
                    df_agg = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                    plt.bar(df_agg.index.astype(str), df_agg.values, color='lightgreen', alpha=0.8)
                    plt.xlabel(cat_col)
                    plt.ylabel(f'Average {num_col}')
                    plt.title(f'Average {num_col} by {cat_col}')
                    plt.xticks(rotation=45)
                else:
                    # Too many categories, show correlation heatmap for numeric columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                        plt.title('Correlation Matrix of Numeric Columns')
                    else:
                        plt.hist(df[num_col].dropna(), bins=15, alpha=0.7, color='lightblue')
                        plt.xlabel(num_col)
                        plt.ylabel('Frequency')
                        plt.title(f'Distribution of {num_col}')
                        
            # Chart 3: Only numeric columns - show correlation or distribution
            else:
                if len(numeric_cols) >= 2:
                    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title(f'{numeric_cols[1]} vs {numeric_cols[0]}')
                    plt.grid(True, alpha=0.3)
                else:
                    plt.hist(df[numeric_cols[0]].dropna(), bins=15, alpha=0.7, color='lightblue')
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {numeric_cols[0]}')
                    
        # Chart 4: Only categorical data
        elif categorical_cols:
            col = categorical_cols[0]
            counts = df[col].value_counts().head(10)  # Top 10 categories
            plt.bar(counts.index.astype(str), counts.values, color='lightcoral', alpha=0.8)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            
        else:
            st.warning("No suitable columns found for automatic chart generation.")
            return None
            
        plt.tight_layout()
        return plt
        
    except Exception as e:
        st.error(f"Error in fallback chart: {e}")
        return None

# ================================
# 🔹 Streamlit UI
# ================================
st.set_page_config(page_title="AI Data Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Assistant with Auto-Visualization")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head())
    
    with col2:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write("**Numeric columns:**", df.select_dtypes(include=[np.number]).columns.tolist())
        st.write("**Categorical columns:**", df.select_dtypes(include=['object']).columns.tolist())

    question = st.text_input("💬 Ask about your data (e.g., 'Show sales by region', 'Distribution of age', 'Correlation between variables')")

    # Initialize session state variables
    if 'answer_text' not in st.session_state:
        st.session_state.answer_text = ""
    if 'chart_path' not in st.session_state:
        st.session_state.chart_path = None
    if 'chart_created' not in st.session_state:
        st.session_state.chart_created = False

    if question and st.button("Analyze", type="primary"):
        with st.spinner("🔍 Analyzing with AI..."):
            data_summary = f"""
            Dataset preview (first 3 rows):
            {df.head(3).to_string(index=False)}
            
            Dataset shape: {df.shape}
            Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
            Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}
            """

            prompt = f"""
            You are a data analyst. Based on this dataset:
            {data_summary}

            Question: {question}

            Please provide:
            1. A clear and concise insight or answer about the data.
            2. If appropriate, suggest a chart type and columns in JSON format like:
               {{"chart_type": "bar", "x": "column_name", "y": "column_name"}}
               
            Available chart types: bar, line, pie, scatter, histogram
            """

            try:
                response = client.chat_completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                )

                answer_text = response.choices[0].message["content"]
                st.session_state.answer_text = answer_text
                
                st.markdown("### 🧠 AI Insight:")
                st.write(answer_text)

                # =======================
                # 🔹 Visualization Logic
                # =======================
                st.markdown("### 📊 Visualization:")
                
                plt_obj = create_auto_chart(df, question, answer_text)
                
                if plt_obj:
                    st.pyplot(plt_obj)
                    
                    # Save chart to temporary file
                    tmp_chart = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    plt_obj.savefig(tmp_chart.name, dpi=300, bbox_inches='tight', facecolor='white')
                    st.session_state.chart_path = tmp_chart.name
                    st.session_state.chart_created = True
                    plt_obj.close()
                    
                    st.success("✅ Chart generated successfully!")
                else:
                    st.info("🤔 No suitable chart could be automatically generated for this data and question.")
                    st.session_state.chart_created = False
                    
            except Exception as e:
                st.error(f"❌ Error during AI analysis: {e}")

    # =======================
    # 🔹 PDF Report Creation
    # =======================
    if st.session_state.answer_text:
        st.markdown("---")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("📝 Creating PDF report..."):
                try:
                    pdf_filename = f"AI_Data_Insights_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(os.getcwd(), pdf_filename)
                    
                    c = canvas.Canvas(pdf_path, pagesize=letter)
                    width, height = letter

                    # Header
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(50, height - 50, "AI Data Insights Report")
                    c.setFont("Helvetica", 10)
                    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    # Dataset Info
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, height - 100, "Dataset Overview:")
                    c.setFont("Helvetica", 10)
                    c.drawString(50, height - 115, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                    c.drawString(50, height - 130, f"File: {uploaded_file.name}")

                    # Question
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, height - 160, "Question:")
                    c.setFont("Helvetica", 10)
                    
                    # Wrap question text
                    y_pos = height - 175
                    words = question.split()
                    line = ""
                    for word in words:
                        test_line = f"{line} {word}".strip()
                        if c.stringWidth(test_line, "Helvetica", 10) < 500:
                            line = test_line
                        else:
                            c.drawString(50, y_pos, line)
                            y_pos -= 15
                            line = word
                    if line:
                        c.drawString(50, y_pos, line)

                    # AI Insight
                    y_pos -= 30
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y_pos, "AI Analysis:")
                    c.setFont("Helvetica", 9)
                    
                    insight_text = st.session_state.answer_text
                    insight_clean = re.sub(r'\{.*?\}', '', insight_text, flags=re.DOTALL)
                    insight_clean = insight_clean.strip()
                    
                    # Wrap insight text
                    y_pos -= 15
                    lines = []
                    words = insight_clean.split()
                    current_line = ""
                    for word in words:
                        test_line = f"{current_line} {word}".strip()
                        if c.stringWidth(test_line, "Helvetica", 9) < 500:
                            current_line = test_line
                        else:
                            lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    
                    for i, line in enumerate(lines[:35]):  # Limit lines
                        if y_pos < 100:
                            c.showPage()
                            y_pos = height - 50
                            c.setFont("Helvetica", 9)
                        
                        c.drawString(50, y_pos, line)
                        y_pos -= 12

                    # Add chart if available
                    if st.session_state.chart_created and st.session_state.chart_path and os.path.exists(st.session_state.chart_path):
                        try:
                            c.showPage()  # New page for chart
                            
                            c.setFont("Helvetica-Bold", 16)
                            c.drawString(100, height - 50, "Data Visualization")
                            
                            # Add image with proper handling
                            img = ImageReader(st.session_state.chart_path)
                            img_width, img_height = img.getSize()
                            
                            # Scale to fit page
                            max_width = 450
                            max_height = 450
                            scale = min(max_width/img_width, max_height/img_height)
                            
                            new_width = img_width * scale
                            new_height = img_height * scale
                            
                            # Center the image
                            x_pos = (width - new_width) / 2
                            y_pos = (height - new_height) / 2 - 20
                            
                            c.drawImage(st.session_state.chart_path, x_pos, y_pos, 
                                      width=new_width, height=new_height,
                                      preserveAspectRatio=True)
                            
                        except Exception as e:
                            st.warning(f"⚠️ Chart could not be added to PDF: {e}")

                    c.save()

                    # Provide download
                    if os.path.exists(pdf_path):
                        st.success("✅ PDF report generated successfully!")
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Report",
                                data=f,
                                file_name=pdf_filename,
                                mime="application/pdf"
                            )
                    else:
                        st.error("❌ PDF report could not be created.")

                except Exception as e:
                    st.error(f"❌ Error creating PDF: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")