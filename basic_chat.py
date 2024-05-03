import os
from openai import OpenAI

# Setup for loading .env
from dotenv import load_dotenv, find_dotenv
my_env = load_dotenv(find_dotenv()) # Read local .env file

# Authenicate
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": '''You are a principal software developer.'''},
        {"role": "system", "content": '''How do I write 'Hello, world!'
                                         in a shell script. No explanations.'''}
    ]
)

print(response.model_dump_json(indent=2))
