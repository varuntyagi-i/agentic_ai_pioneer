from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = LLM(model="gpt-4o-mini")

product_name = "energy drink"
strategist_backstory = "and marketing strategy"


market_researcher = Agent(
    llm=llm,
    role = "Market Researcher",
    goal = "Analyze market trends for the product launch",
    backstory = "Experienced in market trends and consumer behavious analysis",
    tools = [SerperDevTool()],
    verbose = True
)


strategist = Agent(
    llm=llm,
    role = "Product Strategist",
    goal = "Create effective positioning strategies for the product",
    backstory = f"Skilled in competitive positioning {strategist_backstory}",
    verbose = True
)


gather_market_insight_task = Task(
    description=f"Browse the internet to gather insigths on current market trends for the launch of the {product_name} product",
    expected_output=f"List of relevant market trends and consumer preferences, relevant to {product_name}",
    agent=market_researcher
)


develop_positioning_strategy_task = Task(
    description="Based on the market insigts,"
                f"Create a positioning strategy for the {product_name} product, including analysis for the impact and target audience.",
            expected_output="A positioning strategy with target audience and impact notes",
            agent=strategist,
            context=[gather_market_insight_task]
)


crew = Crew(
    agents=[market_researcher, strategist],
    tasks = [gather_market_insight_task, develop_positioning_strategy_task],
    planning=True
)


crew.kickoff()