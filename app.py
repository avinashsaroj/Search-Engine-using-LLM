import gradio as gr
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize search tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Function to handle chatbot responses
def chatbot_response(api_key, user_input, model="Llama3-8b-8192"):
    if not api_key:
        return "⚠️ Please provide your Groq API Key."

    if not user_input.strip():
        return "⚠️ Please enter a question."

    llm = ChatGroq(groq_api_key=api_key, model_name=model)
    tools = [search, arxiv, wiki]

    # Initialize LangChain Agent
    agent = initialize_agent(
        tools, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Run the agent
    response = agent.run(user_input)
    return response

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔎 LangChain – Chat with Search")
    gr.Markdown("Ask anything! The bot searches DuckDuckGo, Wikipedia, and Arxiv to answer.")

    with gr.Row():
        api_key = gr.Textbox(
            label="Enter your Groq API Key",
            type="password",
            placeholder="sk-xxxxxxxx"
        )

    with gr.Row():
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="What is machine learning?",
            lines=2
        )

    submit_btn = gr.Button("Ask")
    output_box = gr.Textbox(label="Response", lines=8)

    submit_btn.click(
        fn=chatbot_response,
        inputs=[api_key, user_input],
        outputs=output_box
    )

# Run app
if __name__ == "__main__":
    demo.launch()
