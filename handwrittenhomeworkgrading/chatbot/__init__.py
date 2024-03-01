# -*- coding: utf-8 -*-

"""Chatbot package for interacting with Azure GPT and providing student chatbot services"""

from .ChatWithAzureGPT import ChatWithAzureGPT
from .StudentChatbot import StudentChatbot

__all__ = [
    "ChatWithAzureGPT",
    "StudentChatbot",
]
