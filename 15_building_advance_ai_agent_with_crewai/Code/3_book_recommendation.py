from crewai import Agent, Crew, Task, Process
from mem0 import MemoryClient
import os

# Set up API key

os.environ['OPENAI_API_KEY'] = ""

#os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

def get_recommendation(user_id: str, user_preferences: str) -> str:
    recommendation_agent = Agent(
        role="Book Recommendation",
        backstory="Ai-powered book advisor who suggests books based on user preferences",
        goal="Provide personalised book recommendations.",
        verbose=True,
    )

    suggest_book_task = Task(
        description=f"Suggest three personalised book recommendations based on user preferences: {user_preferences}.",
        expected_output="A list of three book titles with brief description",
        agent=recommendation_agent
    )

    crew = Crew(
        agents=[recommendation_agent],
        tasks=[suggest_book_task],
        verbose=True,
        process=Process.sequential,
        memory=True,
        memory_config={
            "provider": "mem0",
           # "config": {"user_id": user_id, "api_key": "m0-R5BIOR7mMfo8LViJKAAdt6eF6FwTrKGzeLxRO7F5"},
            "config": {"user_id": user_id, "api_key": "m0-bJ5q6kKFt0pVcJnsFpIMzNHtZXRXlKP1l1QKiVc2"},
            


        }
    )

    result = crew.kickoff()

    print("Final Recommendation:")
    print(result.raw)

    return result.raw


def store_user_recommendation(client: MemoryClient, user_id: str, conversation: list):
    client.add(conversation, user_id=user_id)


if __name__ == '__main__':
    client = MemoryClient(api_key="m0-R5BIOR7mMfo8LViJKAAdt6eF6FwTrKGzeLxRO7F5")

    user_id = input("Enter user ID: ").strip()
    user_preferences = input("Enter preferences: ").strip()

    result = get_recommendation(user_id, user_preferences)

    conversation = [
        {
            "role": "user",
            "content": f"Recommendation for {user_preferences}"
        },
        {
            "role": "assistant",
            "content": result
        }
    ]

    store_user_recommendation(client, user_id, conversation)
