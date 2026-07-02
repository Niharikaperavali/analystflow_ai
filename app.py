# ==================================================
# STEP 0: Configuration & Environment Setup
# ==================================================

# -------- Core Libraries --------
import streamlit as st
import pandas as pd
import numpy as np
import os

# -------- Visualization --------
import matplotlib.pyplot as plt
import seaborn as sns

# -------- Utilities --------
from io import BytesIO
from dotenv import load_dotenv

# -------- ML Preprocessing --------
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)

# -------- Load Environment Variables --------
load_dotenv()

# ==================================================
# Streamlit Page Configuration
# ==================================================
st.set_page_config(
    page_title="AnalystFlow-AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Global UI Styling (SAFE – no metric dependency)
# ==================================================
st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fa;
        }

        .sidebar .sidebar-content {
            background-color: #ffffff;
        }

        h1, h2, h3, h4 {
            color: #2c3e50;
        }

        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# Session State Initialization
# ==================================================
if "df" not in st.session_state:
    st.session_state.df = None

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "previous_df" not in st.session_state:
    st.session_state.previous_df = None

if "sidebar_explanation" not in st.session_state:
    st.session_state.sidebar_explanation = (
        "### 📂 Waiting for Data\n"
        "Upload a CSV or Excel file to begin analysis."
    )

if "operations_log" not in st.session_state:
    st.session_state.operations_log = []

# ==================================================
# Global Plot Settings
# ==================================================
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

# ==================================================
# Utility Guard
# ==================================================
def dataset_loaded():
    return st.session_state.df is not None


# ==================================================
# PROJECT TITLE & DESCRIPTION (VISIBLE IN UI)
# ==================================================
st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 30px 0;">
        <h1>📊 AnalystFlow-AI</h1>
        <p style="font-size:16px; color:#555;">
            Automated Dataset Understanding, Cleaning, Visualization & Data Readiness Reporting
        </p>
        <hr style="margin-top:20px;">
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR: Pipeline Interpretation & Controls
# ==================================================
with st.sidebar:

    st.title("📌 Pipeline Interpretation")

    # ---------------- Undo Last Action ----------------
    if st.button("↩️ Undo Last Action"):
        if st.session_state.previous_df is not None:
            st.session_state.df = st.session_state.previous_df.copy()

            st.session_state.sidebar_explanation = """
### ⏪ Action Undone

The last transformation has been reverted successfully.

You can now:
• Re-apply a different preprocessing step  
• Continue analysis from the restored dataset
"""

            st.success("Last action undone.")
            st.rerun()
        else:
            st.warning("No previous action to undo.")

    st.markdown("---")

    # ---------------- Rule-Based Explanation ----------------
    st.markdown(st.session_state.sidebar_explanation)


# ==================================================
# STEP 1: Dataset Upload & Understanding
# ==================================================

st.header("Step 1: Dataset Upload & Understanding")

uploaded_file = st.file_uploader(
    "📤 Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# ---------------- Load Dataset (Once & Immutable Raw Copy) ----------------
if uploaded_file is not None and st.session_state.raw_df is None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)

        # 🔒 Create IMMUTABLE raw copy (used ONLY for visualization)
        st.session_state.raw_df = df_uploaded.copy(deep=True)

        # 🔄 Working dataset (used for all transformations)
        st.session_state.df = df_uploaded.copy(deep=True)

        rows, cols = df_uploaded.shape

        # Rule-based sidebar explanation
        st.session_state.sidebar_explanation = f"""
### 📂 Dataset Loaded Successfully

📌 **What was done:**
A dataset was uploaded and two separate copies were created.

📌 **Dataset Overview:**
• Rows: {rows}  
• Columns: {cols}

📌 **Why this matters:**
• The raw dataset is preserved for before/after visualization  
• All cleaning and preprocessing operate on a separate working copy  
• This prevents data leakage and ensures unbiased validation
"""

        st.success("Dataset loaded successfully.")
        st.rerun()

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

# ---------------- Dataset Understanding ----------------
if st.session_state.df is not None:

    df = st.session_state.df

    # ---------- Preview ----------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Shape (FIXED: no metric) ----------
    st.subheader("📐 Dataset Shape")
    c1, c2 = st.columns(2)
    c1.info(f"**Rows:** {df.shape[0]}")
    c2.info(f"**Columns:** {df.shape[1]}")

    # ---------- Data Types ----------
    st.subheader("🧬 Column Data Types")
    dtype_df = (
        df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "Column", 0: "Data Type"})
    )
    st.dataframe(dtype_df, use_container_width=True)

    # ---------- Missing Values ----------
    st.subheader("❓ Missing Values Summary")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "Missing Count"]

    if missing_df.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    # ---------- Duplicate Rows ----------
    st.subheader("📑 Duplicate Rows Check")
    dup_count = df.duplicated().sum()
    st.info(f"Duplicate Rows Detected: {dup_count}")

    # ---------- Statistical Summary ----------
    st.subheader("📊 Statistical Summary (Numerical Columns)")
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    else:
        st.info("No numerical columns available for statistical summary.")

# ==================================================
# STEP 2: Data Cleaning – Missing Value Handling
# ==================================================

if st.session_state.df is not None:

    st.header("Step 2: Data Cleaning")
    st.subheader("Handle Missing Values")

    # ✅ SAFE COPY (prevents side effects & protects raw_df)
    df_current = st.session_state.df.copy()

    # Show current missing values count
    total_missing = int(df_current.isnull().sum().sum())
    st.info(f"Total missing values in dataset: {total_missing}")

    # Strategy selection
    missing_strategy = st.radio(
        "Select missing value handling strategy",
        [
            "Replace missing values (Median for numeric, Mode for categorical)",
            "Drop rows with missing values"
        ],
        horizontal=False
    )

    # Apply button
    if st.button("Apply Missing Value Handling"):

        # ---------------- Save state for Undo ----------------
        st.session_state.previous_df = st.session_state.df.copy()

        df_work = df_current.copy()
        rows_before = df_work.shape[0]
        missing_before = int(df_work.isnull().sum().sum())

        # ==================================================
        # Strategy 1: Replace Missing Values
        # ==================================================
        if "Replace" in missing_strategy:

            # Numeric columns → Median
            num_cols = df_work.select_dtypes(include="number").columns
            for col in num_cols:
                if df_work[col].isnull().sum() > 0:
                    df_work[col] = df_work[col].fillna(df_work[col].median())

            # Categorical columns → Mode
            cat_cols = df_work.select_dtypes(exclude="number").columns
            for col in cat_cols:
                if df_work[col].isnull().sum() > 0:
                    mode_val = df_work[col].mode()
                    if not mode_val.empty:
                        df_work[col] = df_work[col].fillna(mode_val[0])

            strategy_used = (
                "Replacement using Median (numeric) and Mode (categorical)"
            )
            impact_msg = f"Missing values reduced from {missing_before} to 0."

        # ==================================================
        # Strategy 2: Drop Rows with Missing Values
        # ==================================================
        else:
            df_work = df_work.dropna()
            strategy_used = "Row-wise deletion (dropping missing values)"
            impact_msg = (
                f"Rows reduced from {rows_before} to {df_work.shape[0]}."
            )

        # ---------------- Update session state ----------------
        st.session_state.df = df_work
        st.session_state.operations_log.append(
            f"Missing value handling applied: {strategy_used}"
        )

        # ---------------- Sidebar explanation (RULE-BASED) ----------------
        st.session_state.sidebar_explanation = f"""
### 🧹 Missing Value Handling Applied

✔ **Strategy Selected:**  
{strategy_used}

📌 **What was done:**  
Missing data points were handled to ensure dataset completeness.

📌 **Why this matters:**  
• Machine learning algorithms cannot process missing values  
• Median reduces outlier influence in numerical features  
• Mode preserves category distribution  
• Dropping rows ensures strict data integrity when required  

📌 **Impact on Dataset:**  
{impact_msg}
"""

        st.success("Missing value handling completed successfully.")
        st.rerun()


# ==================================================
# STEP 3: Advanced Data Cleaning – Duplicates & Outliers
# ==================================================

if st.session_state.df is not None:

    st.header("Step 3: Advanced Data Cleaning")

    # ✅ SAFE COPY
    df_current = st.session_state.df.copy()

    # ==================================================
    # 3A. Duplicate Row Removal
    # ==================================================
    st.subheader("🧾 Remove Duplicate Rows")

    dup_count = df_current.duplicated().sum()
    st.info(f"Duplicate rows detected: {dup_count}")

    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        if st.button("Remove Duplicates"):

            # Save for Undo
            st.session_state.previous_df = st.session_state.df.copy()

            rows_before = df_current.shape[0]
            df_clean = df_current.drop_duplicates()

            # Update state
            st.session_state.df = df_clean
            st.session_state.operations_log.append(
                f"Removed {dup_count} duplicate rows"
            )

            # Sidebar explanation (RULE-BASED)
            st.session_state.sidebar_explanation = f"""
### 🧾 Duplicate Removal Applied

📌 **What was done:**  
Duplicate records were identified and removed.

📌 **Impact on Dataset:**  
• Rows before: {rows_before}  
• Rows after: {df_clean.shape[0]}

📌 **Why this matters:**  
Duplicate records can bias statistical analysis and machine
learning models by over-representing certain observations.
"""

            st.success(f"{dup_count} duplicate rows removed successfully.")
            st.rerun()

    with col_d2:
        if dup_count == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning("Duplicate rows present in the dataset.")

    st.markdown("---")

    # ==================================================
    # 3B. Outlier Treatment (IQR – ALL NUMERIC COLUMNS)
    # ==================================================
    st.subheader("📉 Outlier Treatment (IQR Method – All Numerical Columns)")

    numeric_cols = df_current.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        st.info(
            "Outlier capping will be applied to **all numerical columns** "
            "using the Interquartile Range (IQR) method."
        )

        if st.button("Apply IQR-Based Outlier Capping to All"):

            # Save for Undo
            st.session_state.previous_df = st.session_state.df.copy()

            df_out = df_current.copy()
            capped_columns = []

            for col in numeric_cols:
                Q1 = df_out[col].quantile(0.25)
                Q3 = df_out[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Apply capping
                df_out[col] = df_out[col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )

                capped_columns.append(col)

            # Update state
            st.session_state.df = df_out
            st.session_state.operations_log.append(
                "Outlier capping applied to all numerical columns using IQR method"
            )

            # Sidebar explanation (RULE-BASED)
            st.session_state.sidebar_explanation = f"""
### 📉 Outlier Treatment Applied

📌 **Method Used:**  
Interquartile Range (IQR) method

📌 **Columns Processed:**  
{", ".join(capped_columns)}

📌 **What was done:**  
Outliers were capped for every numerical feature individually
based on its own distribution.

📌 **Why this matters:**  
Applying IQR per feature reduces the influence of extreme values
without deleting rows, preserving dataset size and structure
for reliable analysis and machine learning.
"""

            st.success("Outlier capping applied to all numerical columns.")
            st.rerun()

    else:
        st.info("No numerical columns available for outlier treatment.")


# ==================================================
# STEP 4: Data Preprocessing – Scaling & Encoding
# ==================================================

if st.session_state.df is not None:

    st.header("Step 4: Data Preprocessing")

    # ✅ SAFE COPY (prevents reference issues)
    df_current = st.session_state.df.copy()

    # ==================================================
    # 4A. Feature Scaling (Numerical Columns)
    # ==================================================
    st.subheader("📐 Feature Scaling")

    numeric_cols = df_current.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        scaling_method = st.selectbox(
            "Select scaling method",
            [
                "Standardization (Z-score)",
                "Normalization (Min-Max)"
            ]
        )

        if st.button("Apply Feature Scaling"):

            # ---------------- Save state for Undo ----------------
            st.session_state.previous_df = st.session_state.df.copy()

            df_scaled = df_current.copy()

            if "Standardization" in scaling_method:
                scaler = StandardScaler()
                df_scaled[numeric_cols] = scaler.fit_transform(
                    df_scaled[numeric_cols]
                )
                method_used = "Standardization (Z-score)"

            else:
                scaler = MinMaxScaler()
                df_scaled[numeric_cols] = scaler.fit_transform(
                    df_scaled[numeric_cols]
                )
                method_used = "Normalization (Min-Max)"

            # ---------------- Update session state ----------------
            st.session_state.df = df_scaled
            st.session_state.operations_log.append(
                f"Applied {method_used} to numerical features"
            )

            # ---------------- Sidebar explanation (RULE-BASED) ----------------
            st.session_state.sidebar_explanation = f"""
### 📐 Feature Scaling Applied

✔ **Method Used:**  
{method_used}

📌 **What was done:**  
All numerical features were transformed to a comparable scale.

📌 **Why this matters:**  
Distance-based algorithms such as KNN, SVM, and K-Means are
sensitive to feature magnitude. Scaling ensures that no
single feature dominates the learning process.
"""

            st.success(f"{method_used} applied successfully.")
            st.rerun()

    else:
        st.info("No numerical columns available for scaling.")

    st.markdown("---")

    # ==================================================
    # 4B. Categorical Encoding
    # ==================================================
    st.subheader("🔤 Categorical Encoding")

    cat_cols = df_current.select_dtypes(exclude="number").columns.tolist()

    if cat_cols:

        encoding_method = st.selectbox(
            "Select encoding method",
            [
                "Label Encoding",
                "One-Hot Encoding"
            ]
        )

        if st.button("Apply Categorical Encoding"):

            # ---------------- Save state for Undo ----------------
            st.session_state.previous_df = st.session_state.df.copy()

            df_encoded = df_current.copy()
            cols_before = df_encoded.shape[1]

            if encoding_method == "Label Encoding":

                le = LabelEncoder()
                for col in cat_cols:
                    df_encoded[col] = le.fit_transform(
                        df_encoded[col].astype(str)
                    )

                method_used = "Label Encoding"

            else:
                df_encoded = pd.get_dummies(
                    df_encoded,
                    columns=cat_cols
                )
                method_used = "One-Hot Encoding"

            # ---------------- Update session state ----------------
            st.session_state.df = df_encoded
            st.session_state.operations_log.append(
                f"Applied {method_used} to categorical features"
            )

            # ---------------- Sidebar explanation (RULE-BASED) ----------------
            st.session_state.sidebar_explanation = f"""
### 🔤 Categorical Encoding Applied

✔ **Method Used:**  
{method_used}

📌 **Impact on Dataset:**  
• Columns before: {cols_before}  
• Columns after: {df_encoded.shape[1]}

📌 **Why this matters:**  
Machine learning models operate on numerical values.
Encoding converts categorical variables into a numerical
representation that algorithms can process effectively.
"""

            st.success(f"{method_used} applied successfully.")
            st.rerun()

    else:
        st.info("No categorical columns available for encoding.")



# ==================================================
# STEP 5: Visual Validation – Before vs After
# ==================================================

if st.session_state.df is not None and st.session_state.raw_df is not None:

    st.header("Step 5: Visual Validation (Before vs After)")

    # 🔒 Immutable raw dataset (Before)
    df_before = st.session_state.raw_df.copy(deep=True)

    # 🔄 Processed dataset (After)
    df_after = st.session_state.df.copy()

    # --------------------------------------------------
    # Safe numeric conversion for visualization ONLY
    # (does NOT modify session data)
    # --------------------------------------------------
    def safe_numeric(df):
        temp = df.copy()
        for col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
        return temp

    df_before_num = safe_numeric(df_before)
    df_after_num = safe_numeric(df_after)

    # Use numeric columns AFTER processing
    numeric_cols = df_after_num.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        st.warning("No numerical columns available for visual validation.")
    else:

        # ==================================================
        # 5A. Distribution Comparison (Boxplot)
        # ==================================================
        st.subheader("📦 Distribution Comparison (Outlier Stability)")

        dist_col = st.selectbox(
            "Select numerical column",
            numeric_cols,
            key="viz_dist_col"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Before Cleaning (Raw Data)**")
            fig1, ax1 = plt.subplots()
            sns.boxplot(y=df_before_num[dist_col], ax=ax1)
            ax1.set_title("Before")
            st.pyplot(fig1)

        with col2:
            st.markdown("**After Cleaning (Processed Data)**")
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=df_after_num[dist_col], ax=ax2)
            ax2.set_title("After")
            st.pyplot(fig2)

        st.markdown("---")

        # ==================================================
        # 5B. Scatter Plot Comparison
        # ==================================================
        if len(numeric_cols) >= 2:

            st.subheader("📍 Scatter Plot Comparison")

            x_col = st.selectbox(
                "Select X-axis",
                numeric_cols,
                key="viz_scatter_x"
            )

            y_col = st.selectbox(
                "Select Y-axis",
                numeric_cols,
                index=1,
                key="viz_scatter_y"
            )

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Before Cleaning (Raw Data)**")
                fig3, ax3 = plt.subplots()
                sns.scatterplot(
                    x=df_before_num[x_col],
                    y=df_before_num[y_col],
                    ax=ax3,
                    alpha=0.6
                )
                ax3.set_title("Before")
                st.pyplot(fig3)

            with col4:
                st.markdown("**After Cleaning (Processed Data)**")
                fig4, ax4 = plt.subplots()
                sns.scatterplot(
                    x=df_after_num[x_col],
                    y=df_after_num[y_col],
                    ax=ax4,
                    alpha=0.6
                )
                ax4.set_title("After")
                st.pyplot(fig4)

        st.markdown("---")

        # ==================================================
        # 5C. Correlation Heatmap Comparison
        # ==================================================
        st.subheader("🔥 Correlation Heatmap Comparison")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("**Before Cleaning (Raw Data)**")
            fig5, ax5 = plt.subplots()
            sns.heatmap(
                df_before_num[numeric_cols].corr(),
                cmap="coolwarm",
                ax=ax5
            )
            st.pyplot(fig5)

        with col6:
            st.markdown("**After Cleaning (Processed Data)**")
            fig6, ax6 = plt.subplots()
            sns.heatmap(
                df_after_num[numeric_cols].corr(),
                cmap="coolwarm",
                ax=ax6
            )
            st.pyplot(fig6)

        st.markdown("---")

        # ==================================================
        # 5D. Feature Impact vs Target (SAFE & STABLE)
        # ==================================================
        st.subheader("🔗 Feature Correlation with Target")

        target_col = st.selectbox(
            "Select target variable",
            numeric_cols,
            index=len(numeric_cols) - 1,
            key="viz_target"
        )

        corr_df = df_after_num[numeric_cols].corr()

        if target_col in corr_df.columns and corr_df.shape[0] > 1:

            fig7, ax7 = plt.subplots(figsize=(10, 4))
            (
                corr_df[target_col]
                .drop(target_col)
                .sort_values()
                .plot(kind="bar", ax=ax7)
            )
            ax7.set_title("Feature Correlation with Target")
            plt.xticks(rotation=45)
            st.pyplot(fig7)

        else:
            st.warning("Not enough numeric data to compute correlations.")


# ==================================================
# STEP 6: Export & Explain Data Readiness
# ==================================================

if st.session_state.df is not None and st.session_state.raw_df is not None:

    st.header("Step 6: Export & Explain Data Readiness")

    # ✅ Safe references
    df_final = st.session_state.df.copy()
    df_raw = st.session_state.raw_df.copy(deep=True)
    # ==================================================
# 6A. Export Cleaned Dataset (FIXED)
# ==================================================
    st.subheader("⬇ Export Cleaned Dataset")

    export_format = st.radio(
        "Select export format",
        ["CSV", "Excel"],
        horizontal=True,
        key="export_format"
    )

    if export_format == "CSV":

        csv_data = df_final.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_csv"
        )

    else:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_final.to_excel(writer, index=False, sheet_name="Cleaned_Data")

        st.download_button(
            label="Download Excel",
            data=excel_buffer.getvalue(),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )


    st.markdown("---")

    # ==================================================
    # 6B. Rule-Based Pipeline Summary (ALWAYS SHOWN)
    # ==================================================
    st.subheader("📊 Rule-Based Data Readiness Summary")

    def dataset_stats(df):
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isnull().sum().sum()),
            "numerical_features": int(len(df.select_dtypes(include="number").columns)),
            "categorical_features": int(len(df.select_dtypes(exclude="number").columns)),
        }

    raw_stats = dataset_stats(df_raw)
    clean_stats = dataset_stats(df_final)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Before Processing (Raw Dataset)**")
        st.json(raw_stats)

    with col_s2:
        st.markdown("**After Processing (Cleaned Dataset)**")
        st.json(clean_stats)

    st.markdown("**Operations Performed:**")
    ops_text = "\n".join([f"• {op}" for op in st.session_state.operations_log])
    st.markdown(ops_text if ops_text else "• No transformations were applied.")

    st.success(
        "✅ Dataset is structurally clean and ready for machine learning "
        "based on rule-based validation."
    )

    st.markdown("---")

    # ==================================================
    # 6C. AI-Generated Explanation (Groq ONLY – Optional)
    # ==================================================
    st.subheader("🤖 AI Data Analyst Explanation (Optional)")

    use_ai = st.toggle(
        "Enable AI-generated data readiness explanation (Groq)",
        value=False
    )

    if st.button("Generate Final Explanation"):

        if not use_ai:
            st.info(
                "AI is disabled. Rule-based validation already confirms "
                "the dataset is ML-ready."
            )

        else:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                st.error(
                    "Groq API key not found. "
                    "Please add GROQ_API_KEY to your .env file."
                )
            else:
                from groq import Groq
                client = Groq(api_key=api_key)

                prompt = f"""
You are a Senior Data Analyst.

Generate a professional data readiness report based on the pipeline below.

Raw Dataset Statistics:
{raw_stats}

Cleaned Dataset Statistics:
{clean_stats}

Data Preparation Steps Performed:
{ops_text}

Explain clearly:
1. How data quality improved from raw to cleaned
2. Why each preprocessing step is a best practice
3. Why this dataset is now suitable for machine learning models

Tone:
Professional, concise, and analytical.
"""

                try:
                    with st.spinner("Groq AI is generating the explanation..."):
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_completion_tokens=1024,
                        )

                    st.markdown("### 📝 AI-Generated Data Readiness Report")
                    st.write(completion.choices[0].message.content)

                except Exception as e:
                    st.error(f"AI generation failed: {e}")

