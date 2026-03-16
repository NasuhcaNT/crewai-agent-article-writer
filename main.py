import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Get API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Validate API keys
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing.")
if not serper_api_key:
    raise ValueError("SERPER_API_KEY is missing.")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

# Define the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=True,
    temperature=0.3,
    google_api_key=google_api_key
)

# Define the search tool
search_tool = SerperDevTool()

# Define the agents
researcher = Agent(
    role="Technology Trend Researcher",
    goal="Collect beginner-friendly and up-to-date information about CrewAI and agent architectures.",
    backstory="You are a skilled technology researcher who gathers practical and reliable information from the web.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Medium Technology Content Writer",
    goal='Write an engaging Medium article titled "Agent Architectures: Let’s Build Your AI Team!" based on the research findings.',
    backstory="You are a strong technical writer who can make complex concepts easy to understand.",
    llm=llm,
    verbose=True
)

fact_checker = Agent(
    role="Technology Article Reviewer and Editor",
    goal="Review and improve the article for clarity, correctness, grammar, and flow.",
    backstory="You are a careful editor who prepares technical articles for publication.",
    llm=llm,
    verbose=True
)

# Define the tasks
task_research = Task(
    description="Research CrewAI, agent systems, roles, tools, and setup process with practical examples for beginners.",
    expected_output="A detailed beginner-friendly research summary with examples and code snippets.",
    agent=researcher
)

task_write = Task(
    description='Write a 1200+ word Medium article titled "Agent Architectures: Let’s Build Your AI Team!" using the research notes.',
    expected_output="A complete Medium article draft with introduction, body sections, examples, and conclusion.",
    agent=writer
)

task_fact_check = Task(
    description="Review the article for accuracy, consistency, grammar, and readability. Make it publication-ready.",
    expected_output="A polished final article ready to publish.",
    agent=fact_checker
)

# Build and run the crew
tech_article_crew = Crew(
    agents=[researcher, writer, fact_checker],
    tasks=[task_research, task_write, task_fact_check],
    process=Process.sequential,
    verbose=True
)

print("The article crew is starting...")
result = tech_article_crew.kickoff()

print("\n--- Final Article ---\n")
print(result)