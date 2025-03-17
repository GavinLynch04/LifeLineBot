# LifeLineBot

## Description
A brief description of your project and its purpose.

## Installation
Steps to install and set up the project.

```bash
# Clone the repository
git clone (https://github.com/GavinLynch04/LifeLineBot.git)

# Navigate to the project directory
cd LifeLineBot

# Install dependencies
pip install -r requirements.txt

# Run the main program
python SurivalBot.py

(If this is the first run, the program will take a couple minutes to set up the database of PDFs)

# Chat Mode (default):
Ask any question related to survivial or navigation. The model will default to Gemini API, and when internet access is not avaliable, will use a local hosted model (with a performance hit).

# Search Mode:
Start by typing "search", then hit enter. This will switch to search mode.
Then ask any survial question, and the system will search the database for the most related information.
From there, typing "summarize" will compress the output so it is easier to read and understand.
