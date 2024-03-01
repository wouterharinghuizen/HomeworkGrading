# -*- coding: utf-8 -*-
""" Pipeline for prompts to GPT for grading """

import json
from datetime import datetime
from pathlib import Path
from time import time
from typing import List

import pandas as pd
from tqdm import tqdm

from handwrittenhomeworkgrading.chatbot.ChatWithAzureGPT import ChatWithAzureGPT


class StudentChatbot:
    def __init__(
        self,
        system_message: str,
        dataframe_path: str,
        config_path: Path,
        ratelimit: int = 3600,
    ):
        """
        Initializes the StudentChatbot class with system message, dataframe
        path, and configuration path.

        This method sets up the StudentChatbot by initializing the
        ChatWithAzureGPT instance with the given system message and
        configuration path. It then loads a dataframe from the specified path,
        containing questions, student answers, and correct answers for chatbot
        prompts.

        Parameters
        ----------
        system_message : str
            The initial message to start the conversation.
        dataframe_path : str
            The file path to the CSV containing the homework data.
        config_path : Path
            The file path to the .env configuration file for the Azure GPT.
        """
        self.chatbot = ChatWithAzureGPT(system_message, config_path)
        self._dataframe = pd.read_csv(dataframe_path, sep=",", encoding="utf-8")
        self._prompt = None
        self._prompts = []
        self.set_prompt()
        self.last_api_call_time = 0
        self.ratelimit = ratelimit

    def set_prompt(
        self,
        style_format: str = "Question: {question}\n"
        + "Student Answer: {student_answer}\n"
        + "Correct Answer: {correct_answer}\n",
    ) -> None:
        """
        Creates prompts for the dataset.

        Parameters
        ----------
        style_format : str
            A format string defining how questions and answers should be
            presented to the chatbot. Must include placeholders for question,
            student_answer, and correct_answer.

        Raises
        ------
        ValueError
            If the style_format does not include the required placeholders.
        """
        self._prompt = style_format
        placeholders = ["{question}", "{student_answer}", "{correct_answer}"]
        if not all(p in style_format for p in placeholders):
            raise ValueError(
                "style_format must include placeholders for "
                + "{question}, {student_answer}, and {correct_answer}."
            )
        self._prompts = []
        for _, row in self._dataframe.iterrows():
            prompt = style_format.format(
                question=row["prompt"],
                student_answer=row["output"],
                correct_answer=row["solution"],
            )
            self._prompts.append(prompt)

    def add_few_shot_example(
        self,
        question: str,
        student_answer: str,
        correct_answer: str,
        preferred_output: str,
    ) -> None:
        """
        Adds a few-shot learning example to the chatbot's context, using the
        specified prompt format.

        This method allows users to provide a question, student answer, and
        correct answer to be formatted according to the class's prompt
        structure, along with the preferred response, to guide the chatbot's
        future responses.

        Parameters
        ----------
        question : str
            The question part of the prompt.
        student_answer : str
            The student answer part of the prompt.
        correct_answer : str
            The correct answer part of the prompt.
        preferred_output : str
            The desired output or response for the given prompt configuration.
        """
        # Format the prompt using the provided information and the stored
        # prompt format
        formatted_prompt = self._prompt.format(
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer,
        )

        # Add the formatted prompt as a user message
        self.chatbot.messages.append(
            {"role": "user", "content": formatted_prompt}
        )
        # Add the preferred output as an assistant message
        self.chatbot.messages.append(
            {"role": "assistant", "content": preferred_output}
        )

    def get_responses(self) -> List[str]:
        """
        Generates responses from the chatbot for each prompt in the list of
        prompts.

        This method iterates over the list of formatted prompts, sending each to
        the ChatWithAzureGPT instance for response generation. The responses are
        collected and returned as a list.

        Returns
        -------
        List[str]
            A list of responses from the chatbot for each prompt.
        """
        time_since_last_call = time() - self.last_api_call_time
        if time_since_last_call < self.ratelimit:  # Limit to one call per hour
            raise ValueError(
                "API calls are rate-limited. Please try again later."
            )

        responses = [
            self.chatbot.get_response(prompt) for prompt in tqdm(self._prompts)
        ]
        return responses

    def write_responses_to_dataframe(
        self, responses: List[str], output_path: str = None
    ) -> pd.DataFrame:
        """
        Writes the chatbot responses to the original dataframe, saves it to a
        CSV file in a newly created folder named with the current date and time,
        and creates a JSON file with details about the chat interaction.

        This method updates the dataframe with a new column containing the
        chatbot's responses, creates a directory named after the current date
        and time at the specified output path, saves the dataframe as a CSV file
        within this directory, and also saves a JSON file with information about
        the prompt, system message, and chatbot model type.

        Parameters
        ----------
        responses : List[str]
            A list of responses from the chatbot.
        output_path : str, optional
            The base directory where the new folder will be created and data
            saved.
            If None, defaults to the current directory.
        """

        self._dataframe["response"] = responses

        if output_path is None:
            return self._dataframe

        # Create a folder named with the current date and time
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = Path(output_path) / now
        folder_path.mkdir(parents=True, exist_ok=True)

        # Save the dataframe as a CSV file inside the new folder
        csv_file_path = folder_path / "responses.csv"
        self._dataframe.to_csv(csv_file_path, index=False)

        # Create a JSON file with information about the chat interaction
        info = {
            "prompt_format": self._prompt,
            "system_message": self.chatbot.messages[0]["content"],
        }
        json_file_path = folder_path / "chat_info.json"
        with open(json_file_path, "w") as json_file:
            json.dump(info, json_file, indent=4)

        return self._dataframe
