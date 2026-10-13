from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
import os

# Set up API key
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
# Initialize LLM for natural language processing
llm = LLM(model="gpt-4o-mini")

# User inputs
fitness_goal = "muscle gain"
nutrition_preference = "high protein"

historical_weight_data = ["87kg", "85kg", "88kg", "87kg", "86kg"]

# Fitness Tracker Agent
fitness_tracker_agent = Agent(
    llm=llm,
    role="Fitness Tracker",
    backstory="An AI-powered fitness tracker agent that helps users set goals, track progress, and recommend personalized workout plans.",
    goal="Set fitness goals, track user progress, and provide personalized recommendations based on user preferences.",
    verbose=True,
)

recommendation_agent = Agent(
    llm=llm,
    role="Recommendation Agent",
    backstory="An AI agent that fetches personalized fitness and nutrition recommendations based on user goals and preferences.",
    goal="Search for fitness and nutrition information to provide recommendations that match the user's goals.",
    tools=[SerperDevTool()],
    verbose=True,
)

# Tasks

# 1. Set fitness goals (muscle gain) and track progress
set_and_track_goals = Task(
    description=f"Set the user's fitness goal to {fitness_goal} and track their progress with a "
                f"personalized plan, also take into account the last weight measurements: {historical_weight_data}.",
    expected_output="A fitness plan for muscle gain and progress tracking setup.",
    agent=fitness_tracker_agent,
)

# 2. Fetch personalized fitness and nutrition recommendations based on user preferences (high protein diet)
fetch_fitness_recommendations = Task(
    description=f"Fetch fitness recommendations for goal {fitness_goal} and nutrition preference {nutrition_preference}.",
    expected_output="Personalized fitness and nutrition recommendations based on the user's goal and dietary preference.",
    agent=recommendation_agent,
)

# 3. Provide step-by-step workout plan
provide_workout_plan = Task(
    description="Provide a step-by-step workout plan to help the user achieve their fitness goal.",
    expected_output="A personalized workout plan to meet the muscle gain goal.",
    agent=fitness_tracker_agent,
)

# Crew setup
crew = Crew(
    agents=[fitness_tracker_agent, recommendation_agent],
    tasks=[set_and_track_goals, fetch_fitness_recommendations, provide_workout_plan],
    planning=True,
)

# Start the process
crew.kickoff()