import chainlit as cl

from abc import ABC, abstractmethod


class Persona(ABC):
    @abstractmethod
    async def on_chat_start(self):
        pass

    @abstractmethod
    async def on_message(self, message: cl.Message):
        pass