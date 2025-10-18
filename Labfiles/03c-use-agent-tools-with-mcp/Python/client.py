import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import McpTool, ToolSet, ListSortOrder

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

# Connect to the agents client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
    ),
)

# MCP server configuration
mcp_server_url = "https://learn.microsoft.com/api/mcp"
mcp_server_label = "mslearn"

# Initialize agent MCP tool and toolset
mcp_tool = McpTool(server_label=mcp_server_label, server_url=mcp_server_url)
mcp_tool.set_approval_mode("never")  # no approval prompts

toolset = ToolSet()
toolset.add(mcp_tool)

# ---------- Everything below must remain inside the context ----------
with agents_client:
    # Create a new agent that can use MCP tools
    agent = agents_client.create_agent(
        model=model_deployment,
        name="my-mcp-agent",
        instructions=(
            "You have access to an MCP server called `microsoft.docs.mcp`. "
            "Use the available MCP tools to search Microsoft's official documentation "
            "and answer questions or perform tasks."
        ),
    )
    print(f"Created agent, ID: {agent.id}")
    print(f"MCP Server: {mcp_tool.server_label} at {mcp_tool.server_url}")

    # Create a thread for communication
    thread = agents_client.threads.create()
    print(f"Created thread, ID: {thread.id}")

    # Create a message on the thread
    prompt = input("\nHow can I help?: ")
    message = agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    print(f"Created message, ID: {message.id}")

    # (Optional) set approval mode again explicitly
    mcp_tool.set_approval_mode("never")

    # Create and process agent run in thread with MCP tools
    print("\nProcessing agent thread. Please wait.")
    run = agents_client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
        toolset=toolset,   # <-- give the MCP tools to this run
    )
    print(f"Created run, ID: {run.id}")
    print(f"Run completed with status: {run.status}")
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Display run steps and tool calls (objects, not dicts)
    print("\nRun steps:")
    run_steps = agents_client.run_steps.list(thread_id=thread.id, run_id=run.id)
    for step in run_steps:
        # step is typically a model; access via attributes
        step_id = getattr(step, "id", None)
        step_status = getattr(step, "status", None)
        print(f"  Step {step_id} status: {step_status}")

        details = getattr(step, "step_details", None)
        tool_calls = getattr(details, "tool_calls", []) if details else []
        if tool_calls:
            print("    MCP Tool calls:")
            for call in tool_calls:
                call_id = getattr(call, "id", None)
                call_type = getattr(call, "type", None)
                call_name = getattr(call, "name", None)
                print(f"      id={call_id} type={call_type} name={call_name}")

    # Fetch and log all messages
    print("\nConversation:")
    print("-" * 50)
    messages = agents_client.messages.list(
        thread_id=thread.id, order=ListSortOrder.ASCENDING
    )
    for msg in messages:
        if msg.text_messages:
            last_text = msg.text_messages[-1]
            print(f"{msg.role.upper()}: {last_text.text.value}")
            print("-" * 50)

    # Clean up
    print("Cleaning up agents:")
    agents_client.delete_agent(agent.id)
    print("Deleted agent.")
