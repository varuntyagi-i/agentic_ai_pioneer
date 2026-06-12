from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = LLM(model="gpt-4o-mini")

company_name = "DTDC"
strategist_backstory = "logistics operations"


logistic_analyst = Agent(
    llm=llm,
    role = "Logistic Analyst",
    goal = "Research current state of the logistic operations.",
    backstory = "Experienced in analyzing logistics operations and gathering insights from the market",
    tools = [SerperDevTool()],
    verbose = True
)


optimization_strategist = Agent(
    llm=llm,
    role = "Optimization Strategist",
    goal = "Create optimization strategy for the product",
    backstory = f"Skilled in optimizing the {strategist_backstory}",
    verbose = True
)


gather_logistics_insight_task = Task(
    description=f"Browse the internet to research the current state of the {company_name} logistics operations, focusing on route efficiency and inventory turnover trends",
    expected_output=f"List logistics operations around route and inventory trends, relevant to {company_name}",
    agent=logistic_analyst
)


develop_optimized_strategy_task = Task(
    description="Based on the logistics operations insights, "
            f"provide an optimized strategy for {company_name}'s logistics network.",
    expected_output="An optimized strategy and impact notes",
    agent=optimization_strategist,
    context=[gather_logistics_insight_task]
)


crew = Crew(
    agents=[logistic_analyst, optimization_strategist],
    tasks = [gather_logistics_insight_task, develop_optimized_strategy_task],
    planning=True
)


crew.kickoff()