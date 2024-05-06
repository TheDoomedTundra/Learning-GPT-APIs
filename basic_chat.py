import os
from openai import OpenAI

# Setup for loading .env
from dotenv import load_dotenv, find_dotenv
my_env = load_dotenv(find_dotenv()) # Read local .env file

# Authenicate
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

# Chat completions requires full prompts and roles
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": '''You are a principal software developer.'''},
        {"role": "system", "content": '''How do I write 'Hello, world!'
                                         in a shell script. No explanations.'''}
    ]
)
print(response.model_dump_json(indent=2))

# Test completions response to a specific string
response = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="What is the capital of Canada?",
    max_tokens=256,
    temperature=0
)
print(response.model_dump_json(indent=2))

# Measuring relatedness with embeddings
response = client.embeddings.create(
    input="The cat is sitting on the mat",
    model="text-embedding-ada-002"
)
cat_embeddings = response.data[0].embedding
print(cat_embeddings)

response = client.embeddings.create(
    input="The dog is lying on the rug",
    model="text-embedding-ada-002"
)
dog_embeddings = response.data[0].embedding
print(dog_embeddings)

# Compare the vectors
len(cat_embeddings)

len(dog_embeddings)

# Cosine similarity is a measure of similarity between two non-zero vectors.
# The value can be between 0 and 1; the closer the value is to 1,
# the more similar the vectors are.
import numpy as np
from numpy.linalg import norm

# compute cosine similarity
cosine = np.dot(cat_embeddings,dog_embeddings)/(norm(cat_embeddings)*norm(dog_embeddings)) 
print("Cosine Similarity:", cosine)

## Whisper goes here

# Create image with DALL-E
from IPython.display import Image

response = client.images.generate(
  model="dall-e-2",
  prompt="a rainbow with a pot of gold",
  size="256x256",
  quality="standard",
  n=1, #select the number of images you want generated
)

image_url = response.data[0].url

print(image_url)

Image(url=image_url)

# Can create and manipulate images
