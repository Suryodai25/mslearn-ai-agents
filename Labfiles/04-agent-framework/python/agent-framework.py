import os
import asyncio
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv

# Add references
from agent_framework import AgentThread, ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from pydantic import Field



# ----------------------------
# Tool function (the “plugin”)
# ----------------------------
def send_email(
    to: Annotated[str, Field(description="Who to send the email to")],
    subject: Annotated[str, Field(description="The subject of the email.")],
    body: Annotated[str, Field(description="The text body of the email.")],
):
    # In the lab this just prints, but in real life you’d call Graph/SMTP/etc.
    print("\nTo:", to)
    print("Subject:", subject)
    print(body, "\n")

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
model_deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

async def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load the expenses data file
    script_dir = Path(__file__).parent
    file_path = script_dir / "data.txt"
    with file_path.open("r", encoding="utf-8") as f:
        data = f.read() + "\n"

    # Ask for a prompt
    user_prompt = input(
        f"Here is the expenses data in your file:\n\n{data}\n\n"
        "What would you like me to do with it?\n\n"
    )

    # Run the async agent code
    await process_expenses_data(user_prompt, data)


async def process_expenses_data(prompt: str, expenses_data: str):
    # AzureCliCredential uses your `az login` session; stays fully async.
    async with AzureCliCredential() as credential:
        # AzureAIAgentClient picks up config from env/project (per lab setup)
        async with ChatAgent(
            chat_client=AzureAIAgentClient(async_credential=credential),
            name="expenses_agent",
            instructions=(
                "You are an AI assistant for expense claim submission. "
                "When a user submits expenses data and requests an expense claim, "
                "use the plug-in function to send an email to expenses@contoso.com "
                "with the subject 'Expense Claim' and a body that contains itemized "
                "expenses with a total. Then confirm to the user that you've done so."
            ),
            tools=[send_email],  # tools must be a list
        ) as agent:
            try:
                # Add the input prompt to a list of messages to be submitted
                prompt_messages = [f"{prompt}: {expenses_data}"]
                # Invoke the agent with the messages
                response = await agent.run(prompt_messages)
                # Display the response
                print(f"\n# Agent:\n{response}")
            except Exception as e:
                # Something went wrong
                print(e)


if __name__ == "__main__":
    asyncio.run(main())
