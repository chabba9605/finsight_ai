What os.getenv() does and why you never hardcode keys?
It helps in getting the sercet API keys or other secret passwords from the .env file

What the system/user message structure does?
It helps in defining clear roles for the model and it also helps in getting the desired output

What structured JSON output means and why it is useful for an API?
Structured JSON output is very useful as it helps in furthur organising the data in a better way with clear Key:Data structure you can modify/present or do anything with the data.

What you learned from the Git security incident?
I learned that we should not commit the API keys on to public servers like github and how to reset the git pipeline in case there is a fault in the commit and I also learnt how to exit the terminal text editor mode and also how to make sure that we do not leak any of the password or API keys to public server like Github.

What does the | operator do?
The main work of the | operator is to pass the runnable output of one step into the next step.

What is LCEL and why is it better than raw OpenAI calls?
LCEL stands for Langchain Expression language and it is better because it gives a clean, composable way to build the LLM pipeline, It allows me to add or delete anything from the workflow without re-writing the entire thing.