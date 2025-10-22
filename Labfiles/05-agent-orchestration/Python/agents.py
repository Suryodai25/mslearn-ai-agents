import os
import asyncio
from typing import cast
from dotenv import load_dotenv

from agent_framework import ChatMessage, Role, SequentialBuilder, WorkflowOutputEvent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential  # async credential

# Load env (PROJECT_ENDPOINT from your .env)
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

async def main():
    # Agent instructions
    summarizer_instructions = """
    Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
    Example output:
    App crashes during photo upload.
    User praises dark mode feature.
    """

    classifier_instructions = """
    Classify the feedback as one of the following: Positive, Negative, or Feature request.
    """

    action_instructions = """
    Based on the summary and classification, suggest the next action in one short sentence.
    Example output:
    Escalate as a high-priority bug for the mobile team.
    Log as positive feedback to share with design and marketing.
    Log as enhancement request for product backlog.
    """

    # Create the async credential and chat client INSIDE async context
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(
            async_credential=credential,
            project_endpoint=project_endpoint,  # <-- important
            model_deployment_name=model_deployment,
        ) as chat_client:

            # Create agents
            summarizer = chat_client.create_agent(
                instructions=summarizer_instructions,
                name="summarizer",
            )
            classifier = chat_client.create_agent(
                instructions=classifier_instructions,
                name="classifier",
            )
            action = chat_client.create_agent(
                instructions=action_instructions,
                name="action",
            )

            # The input feedback
            feedback = """
I use the dashboard every day to monitor metrics, and it works well overall.
But when I'm working late at night, the bright screen is really harsh on my eyes.
If you added a dark mode option, it would make the experience much more comfortable.
"""

            # Build sequential orchestration (summarizer -> classifier -> action)
            workflow = SequentialBuilder().participants([summarizer, classifier, action]).build()

            # Run and collect outputs
            outputs: list[list[ChatMessage]] = []
            async for event in workflow.run_stream(f"Customer feedback: {feedback}"):
                if isinstance(event, WorkflowOutputEvent):
                    outputs.append(cast(list[ChatMessage], event.data))

            # Display outputs (last turn)
            if outputs:
                for i, msg in enumerate(outputs[-1], start=1):
                    name = msg.author_name or ("assistant" if msg.role == Role.ASSISTANT else "user")
                    print(f"{'-' * 60}\n{i:02d} [{name}]\n{msg.text}")

if __name__ == "__main__":
    asyncio.run(main())
