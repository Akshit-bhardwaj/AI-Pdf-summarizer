from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("git-cheat-sheet-education.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system",
      "You are the best model to explain and summarize the user data"),
      ("human" , "{data}")]
)


llm = ChatMistralAI(
    model = "mistral-small-latest"
)

prompt = template.format_messages(data = docs[0].page_content)
response = llm.invoke(prompt)
print(response.content)


