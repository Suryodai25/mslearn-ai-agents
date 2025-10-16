import os
from dotenv import load_dotenv
from typing import Any
from pathlib import Path

# Add references
from azure.identity import DefaultAzureCredential   # OK to keep if you run `az login`
# (Or swap to VisualStudioCodeCredential/InteractiveBrowserCredential if you prefer)
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FilePurpose, CodeInterpreterTool, ListSortOrder, MessageRole


def main():

    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load environment variables from .env file
    load_dotenv()
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment:
        raise ValueError("Set PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME in your .env")

    # Path to the data file
    script_dir = Path(__file__).parent
    file_path = script_dir / 'data.txt'
    if not file_path.exists():
        raise FileNotFoundError(f"Expected data file at {file_path}")

    # Show the data being analyzed (optional)
    with file_path.open('r', encoding="utf-8") as f:
        print(f.read() + "\n")

    # ---------------------------
    # OPEN the client and keep it open for the WHOLE flow
    # ---------------------------
    agent_client = AgentsClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        )
    )

    with agent_client:
        # Upload file
        file = agent_client.files.upload_and_poll(
            file_path=file_path, purpose=FilePurpose.AGENTS
        )
        print(f"Uploaded {file.filename}")

        # Code interpreter tool
        code_interpreter = CodeInterpreterTool(file_ids=[file.id])

        # Create agent
        agent = agent_client.create_agent(
            model=model_deployment,
            name="data-agent",
            instructions=(
                "You are an AI agent that analyzes the data in the file that has been uploaded. "
                "Use Python to calculate statistical metrics as necessary."
            ),
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources,
        )
        print(f"Using agent: {agent.name}")

        # Create conversation thread
        thread = agent_client.threads.create()

        # Loop until user types 'quit'
        while True:
            user_prompt = input("Enter a prompt (or type 'quit' to exit): ").strip()
            if user_prompt.lower() == "quit":
                break
            if not user_prompt:
                print("Please enter a prompt.")
                continue

            # Send a prompt to the agent
            agent_client.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_prompt,
            )

            # Run the agent on the thread
            run = agent_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id
            )

            # Check run status
            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                continue

            # Show latest response from the agent
            last_msg = agent_client.messages.get_last_message_text_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT,
            )
            if last_msg:
                print(f"\nLast Message: {last_msg.text.value}\n")

        # After the loop: print conversation history
        print("\nConversation Log:\n")
        messages = agent_client.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.ASCENDING
        )
        for msg in messages:
            if msg.text_messages:
                last = msg.text_messages[-1]
                print(f"{msg.role}: {last.text.value}\n")

        # Clean up
        agent_client.delete_agent(agent.id)
        print("Agent deleted. Goodbye!")


if __name__ == '__main__':
    main()
