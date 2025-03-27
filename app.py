import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

st.set_page_config(page_title="CSV Analyzer AI", layout="wide")
st.title("🧠 Smart CSV/XLS Analyzer")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "OpenAI-key"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = {}

# HELPERS
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def render_data_preview(data):
    st.subheader("📄 Preview Data")
    limit = st.number_input("Show top N rows", 1, len(data), 5)
    st.dataframe(data.head(limit))

def render_chart_controls(data):
    st.subheader("📊 Visualize Columns")
    chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
    chosen_cols = st.multiselect("Choose column(s)", data.columns.tolist())

    if st.button("Generate Chart"):
        if not chosen_cols:
            st.warning("Please select at least one column.")
            return

        fig, ax = plt.subplots()
        if chart_type == "Line":
            data[chosen_cols].plot(ax=ax)
        elif chart_type == "Bar":
            data[chosen_cols].plot(kind="bar", ax=ax)
        elif chart_type == "Scatter" and len(chosen_cols) >= 2:
            data.plot(kind="scatter", x=chosen_cols[0], y=chosen_cols[1], ax=ax)
        elif chart_type == "Histogram":
            data[chosen_cols].plot(kind="hist", bins=20, ax=ax)
        else:
            st.warning("Scatter plot needs two columns.")
            return
        st.pyplot(fig)

def sample_data_for_prompt(data, token_limit=3000):
    header = ",".join(data.columns) + "\n"
    sample = header
    for _, row in data.iterrows():
        row_text = ",".join(map(str, row.values)) + "\n"
        if len(sample + row_text) / 4 > token_limit:
            break
        sample += row_text
    return sample

def ask_ai_about_data(data, user_query):
    csv_sample = sample_data_for_prompt(data)
    full_prompt = (
        f"You're a data expert. Here's a dataset sample:\n\n{csv_sample}\n\nAnswer this question:\n{user_query}"
    )
    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You analyze datasets."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=500,
            temperature=0.5
        )
        return result.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def render_qna_section(data):
    st.subheader("💬 Ask AI about your data")

    history = st.session_state.chat_history
    prev_q = [item['question'] for item in history]
    reuse = st.selectbox("Choose a previous question (optional)", ["New question..."] + prev_q[::-1])
    query = "" if reuse != "New question..." else st.text_input("Type a new question")
    final_query = reuse if reuse != "New question..." else query

    if final_query:
        with st.spinner("Getting answer from AI..."):
            answer = ask_ai_about_data(data, final_query)
            st.success("AI Response:")
            st.write(answer)

            # Save Q&A
            if not any(q['question'] == final_query for q in history):
                history.append({"question": final_query, "answer": answer})

            # Feedback
            st.markdown("### Was the answer helpful?")
            col1, col2 = st.columns(2)
            if col1.button("Yes ✅", key=f"yes-{final_query}"):
                st.session_state.feedback_log[final_query] = "Yes"
            if col2.button("No ❌", key=f"no-{final_query}"):
                st.session_state.feedback_log[final_query] = "No"

            feedback = st.session_state.feedback_log.get(final_query)
            if feedback:
                st.caption(f"Feedback: {feedback}")

def render_sidebar_history():
    with st.sidebar:
        st.header("📚 Chat History")
        for qa in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"_A:_ {qa['answer']}")
            st.markdown("---")

render_sidebar_history()

uploaded_files = st.file_uploader("Upload one or more CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    selected_file = st.selectbox("Select a file to explore", file_names)
    selected_data = next((f for f in uploaded_files if f.name == selected_file), None)

    try:
        df = load_data(selected_data)
        render_data_preview(df)
        render_chart_controls(df)
        render_qna_section(df)
    except Exception as err:
        st.error(f"Could not load file: {err}")
else:
    st.info("Please upload at least one CSV/XLSX file to begin.")

