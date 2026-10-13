from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
import asyncio


class QuestionAnswerPair(BaseModel):
    """Schema for the question and its correct answer."""

    question: str = Field(..., description="The technical question to be asked")
    correct_answer: str = Field(..., description="The correct answer to the question")


# Initialize the search tool
search_tool = SerperDevTool()

# First Crew: Question Preparation
# Create the company research agent
company_researcher = Agent(
    role="Company Research Specialist",
    goal="Gather information about the company and create interview questions with answers",
    backstory="""You are an expert in researching companies and creating technical interview questions.
    You have deep knowledge of tech industry hiring practices and can create relevant
    questions that test both theoretical knowledge and practical skills.""",
    tools=[search_tool],
    verbose=True,
)

# Create the question preparer agent
question_preparer = Agent(
    role="Question and Answer Preparer",
    goal="Prepare comprehensive questions with model answers",
    backstory="""You are an experienced technical interviewer who knows how to create
    challenging yet fair technical questions and provide detailed model answers.
    You understand how to assess different skill levels and create questions that
    test both theoretical knowledge and practical problem-solving abilities.""",
    verbose=True,
)

# Second Crew: Answer Evaluation
# Create the answer evaluator agent
answer_evaluator = Agent(
    role="Answer Evaluator",
    goal="Evaluate if the given answer is correct for the question",
    backstory="""You are a senior technical interviewer who evaluates answers
    against the expected solution. You know how to identify if an answer is
    technically correct and complete.""",
    verbose=True,
)

# Create the follow-up question agent
follow_up_questioner = Agent(
    role="Follow-up Question Specialist",
    goal="Create relevant follow-up questions based on the context",
    backstory="""You are an expert technical interviewer who knows how to create
    meaningful follow-up questions that probe deeper into a candidate's knowledge
    and understanding. You can create questions that build upon previous answers
    and test different aspects of the candidate's technical expertise.""",
    verbose=True,
)


# Create tasks for the first crew
def create_company_research_task(company_name: str, role: str, difficulty: str) -> Task:
    return Task(
        description=f"""Research {company_name} and gather information about:
        1. Their technical interview process
        2. Common interview questions for {role} positions at {difficulty} difficulty level
        3. Technical stack and requirements
        
        Provide a summary of your findings.""",
        expected_output="A report about the company's technical requirements and interview process",
        agent=company_researcher,
    )


def create_question_preparation_task(difficulty: str) -> Task:
    return Task(
        description=f"""Based on the company research, create:
        1. A technical question at {difficulty} difficulty level that tests both theory and practice
        2. A comprehensive model answer that covers all key points
        3. Key points to look for in candidate answers
        
        The question should be appropriate for {difficulty} difficulty level - challenging but fair, and the answer should be detailed.""",
        expected_output="A question and its correct answer",
        output_pydantic=QuestionAnswerPair,
        agent=question_preparer,
    )


# Create task for the second crew
def create_evaluation_task(
    question: str, user_answer: str, correct_answer: str
) -> Task:
    return Task(
        description=f"""Evaluate if the given answer is correct for the question:
        Question: {question}
        Answer: {user_answer}
        Correct Answer: {correct_answer}
        Provide:
        1. Whether the answer is correct (Yes/No)
        2. Key points that were correct or missing
        3. A brief explanation of why the answer is correct or incorrect""",
        expected_output="Evaluation of whether the answer is correct for the question with feedback",
        agent=answer_evaluator,
    )


def create_follow_up_question_task(
    question: str, company_name: str, role: str, difficulty: str
) -> Task:
    return Task(
        description=f"""Based on the following context, create a relevant follow-up question:
        Original Question: {question}
        Company: {company_name}
        Role: {role}
        Difficulty Level: {difficulty}
        
        Create a follow-up question that:
        1. Builds upon the original question
        2. Tests deeper understanding of the topic
        3. Is appropriate for the specified difficulty level
        4. Is relevant to the company and role
        
        The follow-up question should be challenging but fair, and should help
        assess the candidate's technical depth and problem-solving abilities.""",
        expected_output="A follow-up question that builds upon the original question",
        output_pydantic=QuestionAnswerPair,
        agent=follow_up_questioner,
    )


def create_follow_up_crew(
    question: str, company_name: str, role: str, difficulty: str
) -> Crew:
    """Initialize the crew responsible for creating follow-up questions."""
    crew = Crew(
        agents=[follow_up_questioner],
        tasks=[
            create_follow_up_question_task(question, company_name, role, difficulty),
        ],
        process=Process.sequential,
        verbose=True,
    )
    return crew


async def generate_follow_up_question(
    question: str, company_name: str, role: str, difficulty: str
) -> QuestionAnswerPair:
    """Generate a follow-up question asynchronously."""
    result = await create_follow_up_crew(
        question, company_name, role, difficulty
    ).kickoff_async()
    return result.pydantic


# Function to start the interview practice
async def start_interview_practice(
    company_name: str, role: str, difficulty: str = "easy"
):
    # First Crew: Prepare the question and answer
    preparation_crew = Crew(
        agents=[company_researcher, question_preparer],
        tasks=[
            create_company_research_task(company_name, role, difficulty),
            create_question_preparation_task(difficulty),
        ],
        process=Process.sequential,
        verbose=True,
    )

    # Execute the first crew to get the question and model answer
    preparation_result = preparation_crew.kickoff()

    # Generate follow-up question right after preparation (async)

    follow_up_question_task = asyncio.create_task(
        generate_follow_up_question(
            question=preparation_result.pydantic.question,
            company_name=company_name,
            role=role,
            difficulty=difficulty,
        )
    )

    # Print the main question and get user's answer
    print("\nQuestion:")
    print(preparation_result.pydantic.question)
    user_answer = input("\nYour answer: ")

    # Second Crew: Evaluate the answer
    evaluation_crew = Crew(
        agents=[answer_evaluator],
        tasks=[
            create_evaluation_task(
                question=preparation_result.pydantic.question,
                user_answer=user_answer,
                correct_answer=preparation_result.pydantic.correct_answer,
            )
        ],
        process=Process.sequential,
        verbose=True,
    )

    # Execute the second crew and get evaluation
    evaluation_result = evaluation_crew.kickoff()
    print("\nEvaluation:")
    print(evaluation_result)

    input("\nPress Enter to continue to the follow-up question...")

    # Get the follow-up question (it should be ready by now)
    follow_up_question_result = await follow_up_question_task

    # Show the pre-generated follow-up question
    print("\nFollow-up Question:")
    print(follow_up_question_result.question)
    follow_up_answer = input("\nYour answer to the follow-up: ")

    # Evaluate the follow-up answer
    follow_up_evaluation_crew = Crew(
        agents=[answer_evaluator],
        tasks=[
            create_evaluation_task(
                question=follow_up_question_result.question,
                user_answer=follow_up_answer,
                correct_answer=follow_up_question_result.correct_answer,
            )
        ],
        process=Process.sequential,
        verbose=True,
    )

    # Execute the follow-up evaluation
    follow_up_evaluation = follow_up_evaluation_crew.kickoff()
    print("\nFollow-up Evaluation:")
    print(follow_up_evaluation)


if __name__ == "__main__":
    company = "Google"
    role = "Data Scientist"
    print(f"Starting mock interview practice for {role} position at {company}...")
    asyncio.run(start_interview_practice(company, role))


# ------------------------------------------------------------------------------------------------
# For the Streamlit app
# ------------------------------------------------------------------------------------------------
def initialize_preparation_crew(company_name: str, role: str, difficulty: str) -> Crew:
    """Initialize the crew responsible for preparing interview questions."""
    return Crew(
        agents=[company_researcher, question_preparer],
        tasks=[
            create_company_research_task(company_name, role, difficulty),
            create_question_preparation_task(difficulty),
        ],
        process=Process.sequential,
        verbose=True,
    )


def evaluate_answer(question: str, user_answer: str, correct_answer: str) -> str:
    """Create and execute the evaluation crew to assess the user's answer."""
    evaluation_crew = Crew(
        agents=[answer_evaluator],
        tasks=[
            create_evaluation_task(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
            )
        ],
        process=Process.sequential,
        verbose=True,
    )
    return evaluation_crew.kickoff()
