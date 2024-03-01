# -*- coding: utf-8 -*-
"""Connect with the Azure chat"""

import os
from pathlib import Path

import dotenv
import openai


class ChatWithAzureGPT:
    def __init__(self, system_message: str, config_path: Path):
        """
        Initializes the ChatWithAzureGPT class with system message and
        configuration path.

        This method sets up the Azure GPT chatbot by loading the required
        configuration from the specified .env file and initializing the Azure
        OpenAI client with these configurations. It also starts the conversation
        with a system message.

        Parameters
        ----------
        system_message : str
            The initial message from the system to start the conversation.
        config_path : Path
            The file path to the .env configuration file.
        """
        self.config = self.load_configuration(config_path)
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.config["MODEL_ENDPOINT"],
            api_key=self.config["MODEL_API_KEY"],
            api_version=self.config["MODEL_API_VERSION"],
        )
        self.messages = [{"role": "system", "content": system_message}]

    def load_configuration(self, config_path: Path):
        """
        Loads configuration from the specified .env file.

        This method reads environment variables from a .env file located at the
        given path and stores these settings in a configuration dictionary. It
        ensures that all necessary configuration values are present, raising an
        error if any are missing.

        Parameters
        ----------
        config_path : Path
            The file path to the .env configuration file.

        Returns
        -------
        dict
            A dictionary containing the loaded configuration values.

        Raises
        ------
        ValueError
            If any required configuration values are missing.
        """
        dotenv.load_dotenv(config_path)
        config = {
            "MODEL_ENDPOINT": os.getenv("MODEL_ENDPOINT"),
            "MODEL_API_KEY": os.getenv("MODEL_API_KEY"),
            "MODEL_API_VERSION": os.getenv("MODEL_API_VERSION"),
            "MODEL_DEPLOYMENT": os.getenv("MODEL_DEPLOYMENT"),
            "MODEL_TEMPERATURE": os.getenv("MODEL_TEMPERATURE"),
        }
        for key, value in config.items():
            if value is None:
                raise ValueError(f"Configuration for {key} is missing.")
        return config

    def get_response(self, message: str, add_to_history: bool = False) -> str:
        """
        Gets a response from the Azure OpenAI GPT model for the current message.

        This method sends the current user message along with the chat history
        to the Azure OpenAI model to get a response. It can optionally update
        the chat history with the current interaction based on the
        `add_to_history` flag.

        Parameters
        ----------
        message : str
            The user message to get a response for.
        add_to_history : bool
            Flag indicating whether to add the current message and response to
            the chat history.

        Returns
        -------
        str
            The content of the response from the Azure OpenAI GPT model.
        """
        current_message = self.messages + [{"role": "user", "content": message}]
        response = self.client.chat.completions.create(
            model=self.config["MODEL_DEPLOYMENT"],
            messages=current_message,
            temperature=self.config["MODEL_TEMPERATURE"],
            top_p=self.config["TOP_P"],
        )
        response_content = response.choices[0].message.content

        if add_to_history:
            self.messages = current_message + [
                {"role": "assistant", "content": response_content}
            ]

        return response_content
