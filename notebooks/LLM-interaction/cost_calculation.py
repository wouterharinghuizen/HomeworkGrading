# %%
import pandas as pd
from transformers import GPT2TokenizerFast
import os

# %%
# Function to calculate the total cost
def calculate_total_cost(csv_location, system_message, tokenizer, cost_per_1000_tokens):
    # Read the CSV file
    df = pd.read_csv(csv_location)
    
    # Initialize total tokens count
    total_tokens = 0
    
    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Concatenate prompt, output, and solution data
        message_data = str(row['prompt']) + str(row['output']) + str(row['solution'])
        
        # Create message list with system message and user message
        message = [{"role": "system", "content": system_message}]
        message += [{"role": "user", "content": message_data}]

        tokens = tokenizer.tokenize(str(message))
        
        # Count the number of tokens
        total_tokens_datapoint = len(tokens)
        
        # Add to the total tokens count
        total_tokens += total_tokens_datapoint
    
    # Calculate the total cost
    total_cost = total_tokens * 2 * cost_per_1000_tokens / 1000
    
    return total_cost


# %%
COST_PER_1000_TOKENS = 0.03
SYSTEM_MESSAGE = (
    "You are a math teacher on a high school and grade the students with a score from 1 to 5; "
    "You will recieve 3 values: 1) the math question called question. "
    "2) the answer by the student called student_solution. "
    "3) the correct answer called correct_colution."
)

# Initialize GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define the CSV file location
csv_location = os.path.join('..', '..', 'data', 'GHOSTS', 'MATH_symbolicIntegration_14022024.csv')

# Calculate total cost
total_cost = calculate_total_cost(csv_location, SYSTEM_MESSAGE, tokenizer, COST_PER_1000_TOKENS)

# Print the total cost
print(f"Total cost = €{total_cost:.2f}")

# %%



