# Learning-GPT-APIs
Repository for code written while learning the OpenAI APIs

## Setup
`$ pip install openai`

`$ pip install openai[datalib]`

`$ pip install urllib3`

`$ pip install python-dotenv`

`$ pip install tiktoken`

## File Setup
```
import os
from openai import OpenAI

# Setup for loading .env
from dotenv import load_dotenv, find_dotenv
my_env = load_dotenv(find_dotenv()) # Read local .env file

# Authenicate
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)
```
