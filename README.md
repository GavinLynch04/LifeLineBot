# LifeLineBot

## Description
In search and rescue scenarios, victims often lack the knowledge to manage their situation effectively, making external assistance crucial. A lightweight chatbot could provide essential support in such cases, offering guidance in various survival situations. Unlike large language models (LLMs) like ChatGPT, which require internet access and significant storage, this chatbot would be designed to function offline with generalized knowledge to assist individuals in diverse scenarios. Its compact nature ensures it is accessible even without internet connectivity, making it a valuable tool for emergency situations.

## Installation
Steps to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/GavinLynch04/LifeLineBot.git

# Navigate to the project directory
cd LifeLineBot

# Install dependencies
pip install -r requirements.txt

# Setup Gemini API key
# Create .env file, insert free key from Google in the below format
GOOGLE_API_KEY=KEY_HERE

# Run the main program
python SurivalBot.py

# (If this is the first run, the program will take a couple minutes to set up the database of PDFs)
```

## Chat Mode:
Start by typing "chat", then hit enter. This will switch to chat mode.
Ask any question related to survival or navigation. The model will default to Gemini API, 
and when internet access is not available, will use a local hosted model (with a performance hit).

## Search Mode (default):
Start by typing "search", then hit enter. This will switch to search mode.
Then ask any survival question, and the system will search the database for the most related information.
From there, typing "summarize" will compress the output so it is easier to read and understand.
