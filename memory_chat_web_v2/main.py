from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

# 设置你的 OpenAI API Key（部署时设置环境变量 OPENAI_API_KEY）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 初始化模型和记忆
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "history": []})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    reply = conversation.predict(input=user_input)
    history = memory.buffer.split("\n")
    return templates.TemplateResponse("chat.html", {"request": request, "history": history})
