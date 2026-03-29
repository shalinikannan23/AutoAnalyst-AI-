import base64
import streamlit as st

def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("assets/bg2.jpg")

from core.unstructured_engine import unstructured_to_dataframe
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from core.dataset_profiler import profile_dataset
from core.decision_engine import decide_task_type, preprocessing_decisions,get_imputation_options
from core.eda_engine import apply_user_operations

from core.model_engine import generate_model_strategy
from core.explain_engine import compute_feature_scores, generate_feature_insights
from core.rag_engine import (
    build_dataset_knowledge,
    
    build_vector_store,
    ask_llm
)


st.set_page_config(page_title="AUTOANALYST AI", layout="wide")

st.markdown("<h1 style='text-align:center;color:#00E5FF'>🧠 AUTOANALYST AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9FB3C8'>A RAG-Driven Autonomous Data Intelligence System</p>", unsafe_allow_html=True)

# ---------------- SESSION ----------------
for key in ["df", "clean_df", "best_model"]:
    if key not in st.session_state:
        st.session_state[key] = None


st.sidebar.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0F2C, #11162F);
    padding-top: 20px;
}

/* Hide radio circle */
div[role="radiogroup"] input {
    display: none;
}

/* Sidebar title spacing */
div[role="radiogroup"] {
    width: 100%;
}

/* Menu card style */
div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    justify-content: center;

    width: 100%;
    padding: 14px 10px;
    margin: 10px 0px;

    border-radius: 16px;
    background: rgba(255,255,255,0.05);

    font-size: 16px;
    font-weight: 600;
    color: #C7E9FF;

    border: 1.5px solid rgba(0,229,255,0.4);
    cursor: pointer;

    transition: all 0.25s ease-in-out;
}

/* Hover */
div[role="radiogroup"] > label:hover {
    background: rgba(0,229,255,0.15);
    box-shadow: 0 0 15px rgba(0,229,255,0.6);
    transform: translateX(4px);
}

/* Selected pill */
div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, #00E5FF, #4B0082);
    color: white;
    box-shadow: 0 0 20px rgba(75,0,130,0.8);
    border: none;
}

</style>
""", unsafe_allow_html=True)



st.sidebar.markdown("<h1 style='text-align:center;color:#00E5FF'>📂 MODULES</h1>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "UPLOAD DATASET",
        "DATASET PROFILING",
        "DECISION ENGINE",
        "EDA",
        "MODEL TRAINING",
        "EXPLAINABILITY",
        "RAG KNOWLEDGE"
    ]
)


from pandas.errors import EmptyDataError

# ---------------- UPLOAD ----------------
if menu == "UPLOAD DATASET":
    st.subheader("Upload Your Dataset")

    file = st.file_uploader(
        "Upload CSV, Excel, PDF or TXT",
        type=["csv", "xlsx", "xls", "pdf", "txt"]
    )

    if file:
        name = file.name.lower()
        df = None

        # ---- STRUCTURED ----
        if name.endswith(("csv", "xlsx", "xls")):
            try:
                file.seek(0)

                if name.endswith("csv"):
                    try:
                        df = pd.read_csv(file, encoding="utf-8")
                    except UnicodeDecodeError:
                        file.seek(0)
                        df = pd.read_csv(file, encoding="latin1")
                else:
                    df = pd.read_excel(file)

                if df.empty or df.shape[1] == 0:
                    raise EmptyDataError("No columns")

                st.success("Structured dataset loaded")

            except (UnicodeDecodeError, EmptyDataError, pd.errors.ParserError):
                st.warning("File not valid structured data → treating as unstructured")
                file.seek(0)
                df = unstructured_to_dataframe(file)

        # ---- UNSTRUCTURED ----
        else:
            df = unstructured_to_dataframe(file)
            st.warning("Unstructured data detected → converted to structured format")

        st.session_state.df = df
        st.dataframe(df.head())


# ---------------- PROFILER ----------------
elif menu == "DATASET PROFILING":
    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df
        st.subheader("📊 Dataset Profiling")

        profile = profile_dataset(df)

        # ---- Summary Cards ----
        with st.expander("📌 DATASET OVERVIEW", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", profile["rows"])
            col2.metric("Columns", profile["columns"])
            col3.metric("Duplicate Rows", profile["duplicate_rows"])
            col4.metric("Target Column", profile.get("target_column", "N/A"))

        # ---- Missing Values ----
        with st.expander("🟢 MISSING VALUES"):
            missing_df = pd.DataFrame.from_dict(profile["missing_values"], orient="index", columns=["Missing %"])
            fig_missing = px.bar(
                missing_df,
                x=missing_df.index,
                y="Missing %",
                color="Missing %",
                text="Missing %",
                color_continuous_scale="Viridis"
            )
            fig_missing.update_layout(xaxis_title="Column", yaxis_title="Missing Percentage")
            st.plotly_chart(fig_missing, use_container_width=True)

        # ---- Column Types ----
        with st.expander("🟢 COLUMN TYPES"):
            col_types_df = pd.DataFrame.from_dict(profile["column_types"], orient="index", columns=["Type"])
            st.table(col_types_df)

        # ---- Numeric Summary ----
        if "numeric_summary" in profile:
            with st.expander("📈 NUMERIC SUMMARY"):
                numeric_df = pd.DataFrame(profile["numeric_summary"]).T
                st.dataframe(numeric_df)

                # st.markdown("### Boxplots")

                # cols = st.columns(3)  # 3 plots per row
                # i = 0

                # for col in numeric_df.index:
                #     fig, ax = plt.subplots(figsize=(3, 2))
                #     sns.boxplot(x=df[col], ax=ax)
                #     ax.set_title(col, fontsize=9)
                #     ax.tick_params(axis='x', labelsize=7)

                #     cols[i % 3].pyplot(fig)
                #     i += 1


        # ---- Categorical Summary ----
        if "categorical_summary" in profile:
            with st.expander("🟢 CATEGORICAL FEATURES"):
                cat_df = pd.DataFrame(profile["categorical_summary"]).T
                st.table(cat_df)

                for col in profile["categorical_summary"]:
                    st.markdown(f"#### Distribution of {col}")
                    fig_cat = px.histogram(df, x=col, color=col)
                    st.plotly_chart(fig_cat, use_container_width=True)

        # ---- Outliers ----
        if "outliers" in profile:
            with st.expander("⚠ OUTLIERS"):
                outlier_df = pd.DataFrame.from_dict(profile["outliers"], orient="index", columns=["Outliers Count"])
                fig_outlier = px.bar(
                    outlier_df,
                    x=outlier_df.index,
                    y="Outliers Count",
                    color="Outliers Count",
                    text="Outliers Count",
                    color_continuous_scale="Inferno"
                )
                st.plotly_chart(fig_outlier, use_container_width=True)

        
        # ---------------- DECISION ENGINE ----------------
elif menu == "DECISION ENGINE":
    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()
        target = st.selectbox("🎯 Select Target Column", df.columns)

        task = decide_task_type(df, target)

        profile = profile_dataset(df)
        missing = profile["missing_values"]
        outliers = profile["outliers"]

        decisions = preprocessing_decisions(missing, outliers)

        impute_options = get_imputation_options(df, missing)

        st.subheader("🧠 Decision Engine")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📌 Problem Type")
            if task == "classification":
                st.success("Classification Task Detected")
            else:
                st.info("Regression Task Detected")

            st.markdown("### ⚖ Dataset Signals")
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.metric("Missing Columns", sum(v > 0 for v in missing.values()))
            st.metric("Outlier Columns", sum(v > 0 for v in outliers.values()))

        with col2:
            st.markdown("### 🛠 Recommended Preprocessing Steps")

            for d in decisions:
                if "Drop" in d:
                    st.error("🗑 " + d)
                elif "Advanced" in d:
                    st.warning("🧪 " + d)
                elif "Simple" in d:
                    st.info("🧹 " + d)
                elif "outlier" in d.lower():
                    st.warning("📉 " + d)
                else:
                    st.write("➡ " + d)

        st.markdown("---")
        # st.markdown("## 🧪 Choose Imputation Method")

        # if not impute_options:
        #     st.info("✅ No missing values detected. Imputation is not required for this dataset.")
        # else:
        #     impute_col = st.selectbox(
        #         "Select column to impute",
        #         list(impute_options.keys())
        #     )

        #     method = st.radio(
        #         f"Select method for {impute_col}",
        #         impute_options[impute_col]
        #     )

        #     if st.button("Apply Imputation"):
        #         if method == "mean":
        #             df[impute_col] = df[impute_col].fillna(df[impute_col].mean())
        #         elif method == "median":
        #             df[impute_col] = df[impute_col].fillna(df[impute_col].median())
        #         elif method == "mode":
        #             df[impute_col] = df[impute_col].fillna(df[impute_col].mode()[0])
        #         elif method == "drop":
        #             df = df.dropna(subset=[impute_col])

        #         st.session_state.df = df
        #         st.session_state.processed_df = None
        #         st.session_state.operations = []
        #         st.session_state.color_map = None

        #         st.success(f"Imputation applied on column: {impute_col}")
        #         st.dataframe(df.head())



# ---------------- EDA ----------------
# ---------------- EDA ----------------
elif menu == "EDA":
    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()
        st.subheader("🧠 User-Controlled Preprocessing")

        if "operations" not in st.session_state:
            st.session_state.operations = []

        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None

        if "color_map" not in st.session_state:
            st.session_state.color_map = None

        st.markdown("### 📄 Original Dataset")
        st.dataframe(df.head())

        st.markdown("---")
        st.markdown("## ⚙ Build Preprocessing Pipeline")

        operation_type = st.selectbox(
            "Choose Category",
            ["missing", "outlier", "scale", "encode", "datetime", "feature",
             "drop", "text", "balance", "type", "select"]
        )

        column = st.selectbox("Choose Column", df.columns)

        method_map = {
            "missing": ["mean", "median", "mode", "drop", "ffill", "bfill", "constant"],
            "outlier": ["iqr", "zscore", "cap"],
            "scale": ["minmax", "standard", "log"],
            "encode": ["label", "onehot"],
            "datetime": ["parse"],
            "feature": ["extract"],
            "drop": ["column"],
            "text": ["lowercase", "remove_punctuation", "tfidf"],
            "balance": ["undersample", "oversample"],
            "type": ["to_numeric", "to_category"],
            "select": ["correlation", "variance"]
        }

        method = st.selectbox("Choose Method", method_map[operation_type])

        colA, colB, colC = st.columns(3)

        with colA:
            if st.button("➕ Add Step"):
                st.session_state.operations.append(
                    {"type": operation_type, "col": column, "method": method}
                )

        with colB:
            if st.button("↩ Undo Last Step"):
                if st.session_state.operations:
                    st.session_state.operations.pop()

        with colC:
            if st.button("🗑 Clear All Steps"):
                st.session_state.operations = []
                st.session_state.processed_df = None
                st.session_state.color_map = None

        if st.session_state.operations:
            st.markdown("### 🧩 Current Pipeline")

            for i, op in enumerate(st.session_state.operations):
                c1, c2 = st.columns([5,1])
                with c1:
                    st.write(f"{i+1}. {op}")
                with c2:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.operations.pop(i)
                        st.experimental_rerun()

        st.markdown("---")

        if st.button("🚀 Apply Preprocessing"):
            processed_df, color_map = apply_user_operations(
                df, st.session_state.operations
            )

            st.session_state.processed_df = processed_df
            st.session_state.color_map = color_map

        if st.session_state.processed_df is not None:
            st.markdown("### ✅ Processed Dataset (Colored Changes)")

            df_show = st.session_state.processed_df

            if df_show.shape[0] * df_show.shape[1] <= 262144:
                styled = df_show.style.apply(
                    lambda _: st.session_state.color_map, axis=None
                )
                st.dataframe(styled)
            else:
                st.warning("Dataset too large for styling. Showing preview only.")
                st.dataframe(df_show.head(500))

            # 🔽 ALWAYS DOWNLOAD FINAL DATA
            csv = st.session_state.processed_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Preprocessed CSV",
                csv,
                "processed_data.csv",
                "text/csv"
            )


# ---------------- MODEL TRAINING (RECOMMENDATION MODE) ----------------
elif menu == "MODEL TRAINING":
    if st.session_state.processed_df is None:
        st.warning("Run preprocessing first")
    else:
        df = st.session_state.processed_df.copy()

        target = st.selectbox("🎯 Select Target Column", df.columns)

        from core.decision_engine import decide_task_type
        from core.model_engine import generate_model_strategy

        task = decide_task_type(df, target).lower()

        strategy = generate_model_strategy(df, target, task)

        st.subheader("🤖 Model Intelligence Panel")

        c1, c2, c3 = st.columns(3)
        c1.metric("📌 Task Type", strategy["task_type"].capitalize())
        c2.metric("📊 Dataset Size", strategy["dataset_size"])
        c3.metric("🧩 Total Features", strategy["feature_info"]["total_features"])

        st.markdown("---")

        st.markdown("### 🔍 Feature Composition")

        f1, f2 = st.columns(2)
        f1.metric("🔢 Numerical Features", strategy["feature_info"]["num_features"])
        f2.metric("🔤 Categorical Features", strategy["feature_info"]["cat_features"])

        st.markdown("---")

        st.markdown("## 🧠 Recommended Models")

        for model, info in strategy["recommended_models"].items():
            with st.expander(f"📌 {model}", expanded=False):

                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown("**📖 When to use**")
                    st.info(info["when_to_use"])

                with col2:
                    st.markdown("**⚙ Important Hyperparameters**")

                    param_df = pd.DataFrame(
                        info["important_params"].items(),
                        columns=["Parameter", "Suggested Range"]
                    )

                    st.table(param_df)

        st.success("Model strategy generated based on dataset structure and task type.")

# ---------------- EXPLAINABILITY ----------------
elif menu == "EXPLAINABILITY":
    if st.session_state.processed_df is None:
        st.warning("Upload and preprocess dataset first.")
    else:
        df = st.session_state.processed_df.copy()

        st.subheader("📊 Intelligent Feature Explainability Engine")

        target = st.selectbox("Select Target Column", df.columns)

        from core.explain_engine import (
            compute_feature_scores,
            generate_feature_insights,
            find_redundant_features
        )

        score_df = compute_feature_scores(df, target)
        redundant_pairs = find_redundant_features(df.drop(columns=[target]))

        st.markdown("### 🏆 Feature Influence Scores")
        st.dataframe(score_df)

        st.markdown("### 📈 Feature Importance (Top 10)")
        st.bar_chart(score_df["Final_Score"].head(10))

        st.markdown("### 🔍 Inspect a Feature")
        selected_feature = st.selectbox("Choose Feature", score_df.index)

        st.success(score_df.loc[selected_feature, "Explanation"])
        st.metric("Direction", score_df.loc[selected_feature, "Direction"])
        st.metric("Stability", score_df.loc[selected_feature, "Stability"])

        st.markdown("### 🧬 Redundant Features")
        if redundant_pairs:
            for a, b in redundant_pairs:
                st.warning(f"{a} is highly correlated with {b}")
        else:
            st.info("No strongly redundant features detected.")

        st.markdown("### 📝 Dataset Insight")
        insight = generate_feature_insights(score_df, redundant_pairs)
        st.info(insight)


# ---------------- RAG KNOWLEDGE ----------------
elif menu == "RAG KNOWLEDGE":
    if st.session_state.df is None:
        st.warning("Upload dataset first")
    else:
        if st.button("Build Knowledge Base"):
            from core.dataset_profiler import profile_dataset
            from core.decision_engine import decide_task_type
            from core.rag_engine import build_dataset_knowledge, build_vector_store
            from core.eda_engine import outlier_report, correlation_matrix

            df = st.session_state.df

            profile = profile_dataset(df)

            eda = {
                "outliers": outlier_report(df),
                "correlation": correlation_matrix(df)
            }

            target = profile["target_column"]
            task = decide_task_type(df, target)

            texts = build_dataset_knowledge(df, profile, eda, task)
            build_vector_store(texts)

            st.session_state.chat = []
            st.success("Dataset knowledge built successfully!")

        st.subheader("💬 Dataset Intelligence Chat")

        if "chat" not in st.session_state:
            st.session_state.chat = []

        user_input = st.text_input("Ask about project ideas, features, risks, or modeling")

        if user_input:
            from core.rag_engine import ask_llm

            reply = ask_llm(user_input)
            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("Bot", reply))

        for role, msg in st.session_state.chat:
            if role == "You":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 Bot:** {msg}")
