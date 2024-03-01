# %% [markdown]
# # GPT chatting

# %%
import dotenv
import openai
import os
import pandas as pd

from typing import List
from pathlib import Path

import sys

sys.path.append("../../")

from handwrittenhomeworkgrading.chatbot import *

# %%
SYSTEM_MESSAGE = (
    "You are a math teacher on a high school and grade the students with a score from 1 to 5; \n\n"
    "You will recieve 3 values: \n"
    "1. the math question called question. \n"
    "2. the answer by the student called student_answer. \n"
    "3. the correct answer called correct_answer. \n\n"
    "You reponse should be a single symbol integer between 1 and 5 indicating the student score"
)

CONFIG_PATH = Path.cwd().parents[1] / "config" / "local.env"

# %%
chat = ChatWithAzureGPT(SYSTEM_MESSAGE, CONFIG_PATH)

# %%
user_message = '"prompt": "Louis and Jack are sharing a pie.  Louis eats $\\frac{1}{5}$ of the pie, and Jack eats $\\frac{2}{3}$ of the pie.  How much of the pie do Louis and Jack eat altogether?\n Solution: <MISSING> \\boxed{\\frac{13}{15}}.$$\n Fill in the missing steps.", \
        "output": "\n\nFirst find the decimal form of Louiss and Jacks fractions of the pie: $\\frac{1}{5} = 0.2$ and $\\frac{2}{3} = 0.6666...$\nAdd the two decimals together: $0.2 + 0.6666... = 0.8666...$\nTo express this as a fraction, we can find the greatest common denominator of the numerator and denominator, which is 15. So the final answer is $\\frac{13}{15}$.", \
        "Solution": "The denominators $5$ and $3$ have a common multiple of $15$.  We use this to write $\\frac{1}{5}\\cdot \\frac{3}{3}=\\frac{3}{15}$ and $\\frac{2}{3}\\cdot \\frac{5}{5}=\\frac{10}{15}$.  Then, we can add the fractions by adding the numerators and keeping the denominator.  We have $$\\frac{1}{5}+\\frac{2}{3}=\\frac{3}{15}+\\frac{10}{15}=\\frac{3+10}{15}=\\boxed{\\frac{13}{15}}.$$",'

response = chat.get_response(user_message, add_to_history=True)
print(response)

# %%
print(chat.messages)

# %% [markdown]
# # Use dataframe GHOSTS as input

# %%
csv_location = os.path.join(
    "..", "..", "data", "GHOSTS", "MATH_symbolicIntegration_14022024.csv"
)
df = pd.read_csv(csv_location)
df.head(2)

# %%
dataframe_path = os.path.join(
    "..", "..", "data", "GHOSTS", "MATH_symbolicIntegration_14022024.csv"
)

student_chatbot = StudentChatbot(SYSTEM_MESSAGE, dataframe_path, CONFIG_PATH)


# %%
student_chatbot._dataframe = student_chatbot._dataframe.head(1)
student_chatbot.set_prompt(
    "1) question: {question}, \n  2) student_answer: {student_answer}, \n 3) corrent_answer: {correct_answer}."
)


## DO NOT RANDOMLY RUN THIS, IT WILL COST MONEY (IDK HOW MUCH BUT DONT RUN IT BEFORE CHECKING)
# responses = student_chatbot.get_responses()
# new_df = student_chatbot.write_responses_to_dataframe(responses)

# %%
student_chatbot._prompts

# %%
responses = student_chatbot.get_responses()

# %%
df_with_reponses = student_chatbot.write_responses_to_dataframe(responses)

# %%
df_with_reponses.iloc[0]["response"]
