import os
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Set your Google AI Studio API key
API_KEY = "your-api-key-here"  # Replace with your actual API key from Google AI Studio
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# Specify the Gemma 3 model (adjust based on availability in Google AI Studio)
model_name = "gemma-3-27b-it"  # Hypothetical; verify exact name in Google AI Studio

# Function to call Gemma 3 via Google GenAI SDK
def call_gemma_3(prompt):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

# Wrap Gemma 3 in a LangChain Runnable for CrewAI compatibility
gemma_runnable = RunnableLambda(call_gemma_3)

# Define Agents
researcher = Agent(
    role="Researcher",
    goal="Gather detailed information on the specified topic.",
    backstory="You are an expert researcher skilled at finding and summarizing information.",
    verbose=True,
    allow_delegation=False,
    llm=gemma_runnable  # Using Gemma 3 as the reasoning engine
)

analyst = Agent(
    role="Analyst",
    goal="Analyze research data and extract key insights.",
    backstory="You are a data analyst with a knack for identifying trends and key points.",
    verbose=True,
    allow_delegation=False,
    llm=gemma_runnable
)

writer = Agent(
    role="Writer",
    goal="Compile research and analysis into a concise, well-written report.",
    backstory="You are a skilled writer who crafts clear and engaging reports.",
    verbose=True,
    allow_delegation=False,
    llm=gemma_runnable
)

# Define Tasks
def create_research_task(topic):
    return Task(
        description=f"Research the topic: '{topic}'. Provide a detailed summary of relevant information.",
        agent=researcher,
        expected_output="A detailed summary of the topic based on available knowledge."
    )

def create_analysis_task():
    return Task(
        description="Analyze the research summary and extract 3-5 key insights or trends.",
        agent=analyst,
        expected_output="A list of 3-5 key insights derived from the research summary."
    )

def create_writing_task():
    return Task(
        description="Compile the research summary and insights into a concise report.",
        agent=writer,
        expected_output="A well-written report summarizing the research and insights."
    )

# Main function to run the crew
def run_research_crew(topic):
    # Create tasks
    research_task = create_research_task(topic)
    analysis_task = create_analysis_task()
    writing_task = create_writing_task()

    # Assemble the crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # Tasks executed one after another
        verbose=2  # Detailed logging
    )

    # Kick off the crew
    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    research_topic = input("Enter a research topic: ")  # e.g., "Impact of AI on healthcare"
    final_report = run_research_crew(research_topic)
    print("\nFinal Report:")
    print(final_report)