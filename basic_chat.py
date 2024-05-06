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


# Generate data and fine tune the model
import pandas as pd
import random

# lists to hold the random prompt values
locations = ['the moon', 'a space ship', 'in outer space']
alien_types = ["Grey","Reptilian","Nordic","Shape shifting"]
hero_goals = ["save the Earth", "destroy the alien home planet", "save the human race"]

# prompt template to be completed using values from the lists above
prompt = ''' Imagine the plot for a new science fiction movie. The location is {location}. Humans
               are fighting the {alien_type} aliens. The hero of the movie intends to {hero_goal}. 
               Write the movie plot in 50 words or less. '''

sub_prompt = "{location}, {alien_type}, {hero_goal}"

df = pd.DataFrame()

# To fine-tune a model, you are required to provide at least 10 examples. 
# You'll see improvements from fine-tuning on 50 to 100 training examples 
for i in range(100): 
    
    # retrieve random numbers based on the length of the lists
    location = random.randint(0,len(locations)-1)
    alien_type = random.randint(0,len(alien_types)-1)
    hero_goal = random.randint(0,len(hero_goals)-1)
    
    # use the prompt template and fill in the values
    model_prompt = prompt.format(location=locations[location], alien_type=alien_types[alien_type], 
                           hero_goal=hero_goals[hero_goal])
    
    # track the values used to fill in the template
    model_sub_prompt = sub_prompt.format(location=locations[location], alien_type=alien_types[alien_type], 
                           hero_goal=hero_goals[hero_goal])

    # retrieve a model generated movie plot based on the input prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
           {"role": "system", "content": '''You help write movie scripts.'''},
           {"role": "user", "content": model_prompt}
        ],
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # retrieve the finish reason for the model
    finish_reason = response.choices[0].finish_reason
    
    # retrieve the response 
    response_txt = response.choices[0].message.content
    
    # add response, prompt, etc. to a DataFrame
    new_row = {
        'location'
        'alien_type'
        'hero_goal'
        'prompt':model_prompt, 
        'sub_prompt':model_sub_prompt, 
        'response_txt':response_txt, 
        'finish_reason':finish_reason}
    
    new_row = pd.DataFrame([new_row])
    
    df = pd.concat([df, new_row], axis=0, ignore_index=True)

#save DataFrame to a CSV 
df.to_csv("science_fiction_plots.csv")
