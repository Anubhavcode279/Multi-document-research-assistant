import streamlit as st, requests

st.title("Multi-document research assistant")
q = st.text_area("Ask a research question")
if st.button("Ask") and q:
    with st.spinner("Researching (local model, this can take a minute or two)..."):
        r = requests.post("http://localhost:8000/ask", json={"question": q}, timeout=900).json()
    st.markdown(r["answer"])
    with st.expander("How the agent worked"):
        st.write("Sub-questions:", r["sub_questions"])
        st.write("Self-check verdict:", r["check"])
        st.write("Revisions:", r["revisions"])
    with st.expander("Sources used"):
        st.write(r["sources"])