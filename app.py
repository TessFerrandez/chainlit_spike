import chainlit as cl

from personas.chatty_mcchatface import ChattyMcChatface
from personas.doc_summarizer import DocSummarizer
from personas.dr_ragilicious import DrRAGilicious
from personas.mr_andersson import MrAndersson

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Chatty McChatface",
            markdown_description="I'll chat with you about anything you want!",
        ),
        cl.ChatProfile(
            name="Doc Summarizer",
            markdown_description="I'll summarize all your docz",
        ),
        cl.ChatProfile(
            name="Dr. RAGilicious",
            markdown_description="Send me your docs and we'll chat about them",
        ),
        cl.ChatProfile(
            name="Mr. Andersson",
            markdown_description="I'm the agent, I can do maths",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "Chatty McChatface":
        persona = ChattyMcChatface()
    elif chat_profile == "Doc Summarizer":
        persona = DocSummarizer()
    elif chat_profile == "Dr. RAGilicious":
        persona = DrRAGilicious()
    elif chat_profile == "Mr. Andersson":
        persona = MrAndersson()

    if persona:
        await persona.on_chat_start()
        cl.user_session.set("persona", persona)
    else:
        await cl.Message(content=f"starting chat using the {chat_profile} chat profile").send()


@cl.on_message
async def on_message(message: cl.Message):
    persona = cl.user_session.get("persona")
    if persona:
        await persona.on_message(message)
    else:
        await cl.Message(content=f"received message in {chat_profile} chat profile").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)