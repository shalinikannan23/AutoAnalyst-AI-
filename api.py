# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt

# from core.dataset_profiler import profile_dataset
# from core.decision_engine import decide_task_type, preprocessing_decisions,get_imputation_options
# from core.eda_engine import apply_user_operations

# from core.model_engine import generate_model_strategy
# from core.explain_engine import compute_feature_scores, generate_feature_insights
# from core.rag_engine import (
#     build_dataset_knowledge,
#     build_vector_store,
#     ask_llm
# )

# st.set_page_config(page_title="AUTOANALYST AI", layout="wide")
# st.markdown("<h1 style='text-align:center;color:#4B0082'>🧠 AUTOANALYST AI</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center;color:gray'>Autonomous Data Intelligence System</p>", unsafe_allow_html=True)

# # ---------------- SESSION ----------------
# for key in ["df", "clean_df", "best_model"]:
#     if key not in st.session_state:
#         st.session_state[key] = None

# st.sidebar.markdown("""
# <style>
# div[role="radiogroup"] > label {
#     background: #1f1f2e;
#     padding: 14px;
#     margin: 10px 0px;
#     border-radius: 12px;
#     font-size: 18px;
#     font-weight: 700;
#     text-align: center;
#     color: white;
#     border: 2px solid #4B0082;
#     transition: all 0.2s ease-in-out;
# }
# div[role="radiogroup"] > label:hover {
#     background: #4B0082;
#     transform: scale(1.03);
# }
# </style>
# """, unsafe_allow_html=True)

# st.sidebar.markdown("<h2 style='text-align:center;color:#4B0082'>📂 MODULES</h2>", unsafe_allow_html=True)

# menu = st.sidebar.radio(
#     "",
#     [
#         "UPLOAD DATASET",
#         "DATASET PROFILING",
#         "DECISION ENGINE",
#         "EDA",
#         "MODEL TRAINING",
#         "EXPLAINABILITY",
#         "RAG KNOWLEDGE"
#     ]
# )


# # ---------------- UPLOAD ----------------
# if menu == "UPLOAD DATASET":
#     st.subheader("Upload Your Dataset")
#     file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
#     if file:
#         df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
#         st.session_state.df = df
#         st.success("Dataset Loaded Successfully!")
#         st.dataframe(df.head())

# # ---------------- PROFILER ----------------
# elif menu == "DATASET PROFILING":
#     if st.session_state.df is None:
#         st.warning("Upload dataset first")
#     else:
#         df = st.session_state.df
#         st.subheader("📊 Dataset Profiling")

#         profile = profile_dataset(df)

#         # ---- Summary Cards ----
#         with st.expander("📌 DATASET OVERVIEW", expanded=True):
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("Rows", profile["rows"])
#             col2.metric("Columns", profile["columns"])
#             col3.metric("Duplicate Rows", profile["duplicate_rows"])
#             col4.metric("Target Column", profile.get("target_column", "N/A"))

#         # ---- Missing Values ----
#         with st.expander("🟢 MISSING VALUES"):
#             missing_df = pd.DataFrame.from_dict(profile["missing_values"], orient="index", columns=["Missing %"])
#             fig_missing = px.bar(
#                 missing_df,
#                 x=missing_df.index,
#                 y="Missing %",
#                 color="Missing %",
#                 text="Missing %",
#                 color_continuous_scale="Viridis"
#             )
#             fig_missing.update_layout(xaxis_title="Column", yaxis_title="Missing Percentage")
#             st.plotly_chart(fig_missing, use_container_width=True)

#         # ---- Column Types ----
#         with st.expander("🟢 COLUMN TYPES"):
#             col_types_df = pd.DataFrame.from_dict(profile["column_types"], orient="index", columns=["Type"])
#             st.table(col_types_df)

#         # ---- Numeric Summary ----
#         if "numeric_summary" in profile:
#             with st.expander("📈 NUMERIC SUMMARY"):
#                 numeric_df = pd.DataFrame(profile["numeric_summary"]).T
#                 st.dataframe(numeric_df)

#                 # st.markdown("### Boxplots")

#                 # cols = st.columns(3)  # 3 plots per row
#                 # i = 0

#                 # for col in numeric_df.index:
#                 #     fig, ax = plt.subplots(figsize=(3, 2))
#                 #     sns.boxplot(x=df[col], ax=ax)
#                 #     ax.set_title(col, fontsize=9)
#                 #     ax.tick_params(axis='x', labelsize=7)

#                 #     cols[i % 3].pyplot(fig)
#                 #     i += 1


#         # ---- Categorical Summary ----
#         if "categorical_summary" in profile:
#             with st.expander("🟢 CATEGORICAL FEATURES"):
#                 cat_df = pd.DataFrame(profile["categorical_summary"]).T
#                 st.table(cat_df)

#                 for col in profile["categorical_summary"]:
#                     st.markdown(f"#### Distribution of {col}")
#                     fig_cat = px.histogram(df, x=col, color=col)
#                     st.plotly_chart(fig_cat, use_container_width=True)

#         # ---- Outliers ----
#         if "outliers" in profile:
#             with st.expander("⚠ OUTLIERS"):
#                 outlier_df = pd.DataFrame.from_dict(profile["outliers"], orient="index", columns=["Outliers Count"])
#                 fig_outlier = px.bar(
#                     outlier_df,
#                     x=outlier_df.index,
#                     y="Outliers Count",
#                     color="Outliers Count",
#                     text="Outliers Count",
#                     color_continuous_scale="Inferno"
#                 )
#                 st.plotly_chart(fig_outlier, use_container_width=True)

        
#         # ---------------- DECISION ENGINE ----------------
# elif menu == "DECISION ENGINE":
#     if st.session_state.df is None:
#         st.warning("Upload dataset first")
#     else:
#         df = st.session_state.df.copy()
#         target = st.selectbox("🎯 Select Target Column", df.columns)

#         task = decide_task_type(df, target)

#         missing = missing_value_report(df)
#         outliers = outlier_report(df)

#         decisions = preprocessing_decisions(missing, outliers)
#         impute_options = get_imputation_options(df, missing)

#         st.subheader("🧠 Decision Engine")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("### 📌 Problem Type")
#             if task == "classification":
#                 st.success("Classification Task Detected")
#             else:
#                 st.info("Regression Task Detected")

#             st.markdown("### ⚖ Dataset Signals")
#             st.metric("Rows", df.shape[0])
#             st.metric("Columns", df.shape[1])
#             st.metric("Missing Columns", sum(v > 0 for v in missing.values()))
#             st.metric("Outlier Columns", sum(v > 0 for v in outliers.values()))

#         with col2:
#             st.markdown("### 🛠 Recommended Preprocessing Steps")

#             for d in decisions:
#                 if "Drop" in d:
#                     st.error("🗑 " + d)
#                 elif "Advanced" in d:
#                     st.warning("🧪 " + d)
#                 elif "Simple" in d:
#                     st.info("🧹 " + d)
#                 elif "outlier" in d.lower():
#                     st.warning("📉 " + d)
#                 else:
#                     st.write("➡ " + d)

#         st.markdown("---")
#         st.markdown("## 🧪 Choose Imputation Method")

#         user_choices = {}

#         for col, methods in impute_options.items():
#             st.markdown(f"**Column: {col}**")

#             if "mean" in methods:
#                 st.write("Mean:", round(df[col].mean(), 3))
#                 st.write("Median:", round(df[col].median(), 3))
#                 st.write("Mode:", df[col].mode()[0])
#             else:
#                 st.write("Mode:", df[col].mode()[0])

#             choice = st.radio(
#                 f"Select method for {col}",
#                 methods,
#                 key=col
#             )

#             user_choices[col] = choice

#         if st.button("Apply Imputation"):
#             for col, method in user_choices.items():
#                 if method == "mean":
#                     df[col].fillna(df[col].mean(), inplace=True)
#                 elif method == "median":
#                     df[col].fillna(df[col].median(), inplace=True)
#                 elif method == "mode":
#                     df[col].fillna(df[col].mode()[0], inplace=True)

#             st.session_state.df = df
#             st.success("Imputation applied based on your selections!")
#             st.dataframe(df.head())

# # ---------------- EDA ----------------
# # ---------------- EDA ----------------
# elif menu == "EDA":
#     if st.session_state.df is None:
#         st.warning("Upload dataset first")
#     else:
#         df = st.session_state.df
#         st.subheader("🧠 User-Controlled Preprocessing")

#         if "operations" not in st.session_state:
#             st.session_state.operations = []

#         if "processed_df" not in st.session_state:
#             st.session_state.processed_df = None

#         if "color_map" not in st.session_state:
#             st.session_state.color_map = None

#         st.markdown("### 📄 Original Dataset")
#         st.dataframe(df.head())

#         st.markdown("---")

#         st.markdown("## ⚙ Build Preprocessing Pipeline")

#         operation_type = st.selectbox(
#             "Choose Category",
#             ["missing", "outlier", "scale", "encode", "datetime", "feature"]
#         )

#         column = st.selectbox("Choose Column", df.columns)

#         method_map = {
#             "missing": ["mean", "median", "mode", "drop", "ffill", "bfill", "constant"],
#             "outlier": ["iqr", "zscore", "cap"],
#             "scale": ["minmax", "standard", "log"],
#             "encode": ["label", "onehot"],
#             "datetime": ["parse"],
#             "feature": ["extract"]
#         }

#         method = st.selectbox("Choose Method", method_map[operation_type])

#         colA, colB, colC = st.columns(3)

#         with colA:
#             if st.button("➕ Add Step"):
#                 st.session_state.operations.append(
#                     {"type": operation_type, "col": column, "method": method}
#                 )

#         with colB:
#             if st.button("↩ Undo Last Step"):
#                 if st.session_state.operations:
#                     st.session_state.operations.pop()

#         with colC:
#             if st.button("🗑 Clear All Steps"):
#                 st.session_state.operations = []
#                 st.session_state.processed_df = None
#                 st.session_state.color_map = None

#         if st.session_state.operations:
#             st.markdown("### 🧩 Current Pipeline")

#             for i, op in enumerate(st.session_state.operations):
#                 c1, c2 = st.columns([5,1])
#                 with c1:
#                     st.write(f"{i+1}. {op}")
#                 with c2:
#                     if st.button("❌", key=f"remove_{i}"):
#                         st.session_state.operations.pop(i)
#                         st.experimental_rerun()

#         st.markdown("---")

#         if st.button("🚀 Apply Preprocessing"):
#             processed_df, color_map = apply_user_operations(
#                 df, st.session_state.operations
#             )

#             st.session_state.processed_df = processed_df
#             st.session_state.color_map = color_map

#         if st.session_state.processed_df is not None:
#             st.markdown("### ✅ Processed Dataset (Colored Changes)")

#             styled = st.session_state.processed_df.style.apply(
#                 lambda _: st.session_state.color_map, axis=None
#             )

#             st.dataframe(styled)

#             csv = st.session_state.processed_df.to_csv(index=False).encode("utf-8")

#             st.download_button(
#                 "⬇ Download Preprocessed CSV",
#                 csv,
#                 "processed_data.csv",
#                 "text/csv"
#             )

# # ---------------- MODEL TRAINING (RECOMMENDATION MODE) ----------------
# elif menu == "MODEL TRAINING":
#     if st.session_state.clean_df is None:
#         st.warning("Run EDA first")
#     else:
#         df = st.session_state.clean_df
#         target = st.selectbox("🎯 Select Target Column", df.columns)

#         task = decide_task_type(df, target).lower()

#         from core.model_engine import generate_model_strategy
#         strategy = generate_model_strategy(df, target, task)

#         st.subheader("🤖 Model Intelligence Panel")

#         # ====== TOP METRICS ======
#         c1, c2, c3 = st.columns(3)
#         c1.metric("📌 Task Type", strategy["task_type"].capitalize())
#         c2.metric("📊 Dataset Size", strategy["dataset_size"])
#         c3.metric("🧩 Total Features", strategy["feature_info"]["total_features"])

#         st.markdown("---")

#         # ====== FEATURE BREAKDOWN ======
#         st.markdown("### 🔍 Feature Composition")

#         f1, f2 = st.columns(2)
#         f1.metric("🔢 Numerical Features", strategy["feature_info"]["num_features"])
#         f2.metric("🔤 Categorical Features", strategy["feature_info"]["cat_features"])

#         st.markdown("---")

#         # ====== MODEL RECOMMENDATIONS ======
#         st.markdown("## 🧠 Recommended Models")

#         for model, info in strategy["recommended_models"].items():
#             with st.expander(f"📌 {model}", expanded=False):

#                 col1, col2 = st.columns([2, 3])

#                 with col1:
#                     st.markdown("**📖 When to use**")
#                     st.info(info["when_to_use"])

#                 with col2:
#                     st.markdown("**⚙ Important Hyperparameters**")

#                     param_df = pd.DataFrame(
#                         info["important_params"].items(),
#                         columns=["Parameter", "Suggested Range"]
#                     )

#                     st.table(param_df)

#         st.markdown("---")

#         st.success("Model strategy generated based on dataset structure and task type.")

# # ---------------- EXPLAINABILITY ----------------
# elif menu == "EXPLAINABILITY":
#     if st.session_state.clean_df is None:
#         st.warning("Upload and clean dataset first.")
#     else:
#         df = st.session_state.clean_df

#         st.subheader("📊 Feature Scoring & Dataset Explainability")

#         target = st.selectbox("Select Target Column", df.columns)

#         from core.explain_engine import compute_feature_scores, generate_feature_insights

#         score_df = compute_feature_scores(df, target)

#         st.markdown("### 🏆 Feature Importance Scores")
#         st.dataframe(score_df)

#         st.markdown("### 📈 Top Feature Contributions")
#         st.bar_chart(score_df["Final_Score"].head(10))

#         st.markdown("### 📝 Dataset Insight")
#         insight = generate_feature_insights(score_df)
#         st.info(insight)

# # ---------------- RAG KNOWLEDGE ----------------
# elif menu == "RAG KNOWLEDGE":
#     if st.session_state.df is None:
#         st.warning("Upload dataset first")
#     else:
#         if st.button("Build Knowledge Base"):
#             profile = profile_dataset(st.session_state.df)

#             eda = {
#                 "outliers": outlier_report(st.session_state.df),
#                 "correlation": correlation_matrix(st.session_state.df)
#             }

#             target = profile["target_column"]
#             task = decide_task_type(st.session_state.df, target)

#             texts = build_dataset_knowledge(profile, eda, task)
#             build_vector_store(texts)

#             st.success("Dataset knowledge built successfully!")

#         st.subheader("💬 Dataset Intelligence Chat")

#         if "chat" not in st.session_state:
#             st.session_state.chat = []

#         user_input = st.text_input("Ask about project ideas, innovations, or dataset usage")

#         if user_input:
#             reply = ask_llm(user_input)

#             st.session_state.chat.append(("You", user_input))
#             st.session_state.chat.append(("Bot", reply))

#         for role, msg in st.session_state.chat:
#             if role == "You":
#                 st.markdown(f"**🧑 You:** {msg}")
#             else:
#                 st.markdown(f"**🤖 Bot:** {msg}")

