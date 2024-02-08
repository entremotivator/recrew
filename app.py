import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama

# Define the search tool
search_tool = DuckDuckGoSearchRun()

# Load Human Tools
human_tools = load_tools(["human"])

# Custom tool for reading webpage content
class ContentTools:
    @tool("Read webpage content")
    def read_content(url: str) -> str:
        """Read content from a webpage."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text()
        return text_content[:5000]

# Define Streamlit app title and sidebar
st.title("Real Estate Investment Property Research")
st.sidebar.title("Property Analysis")

# Define the address of interest
address = st.sidebar.text_input("Enter address:", "123 Main St, Anytown, USA")

# Create agents
manager = Agent(
    role='Project Manager',
    goal='Coordinate the project to ensure a seamless integration of research findings into investment strategies',
    verbose=True,
    backstory="""With a strategic mindset and a knack for leadership, you excel at guiding teams towards
    their goals, ensuring projects not only meet but exceed expectations.""",
    allow_delegation=True,
    max_iter=10,
    max_rpm=20,
)

researcher = Agent(
    role='Real Estate Researcher',
    goal=f'Analyze the investment potential of properties around {address}',
    verbose=True,
    backstory="""With a keen eye for market trends, you are dedicated to analyzing real estate data 
    to identify promising investment opportunities."""
)

analyst = Agent(
    role='Data Analyst',
    goal=f'Analyze market trends and data for {address}',
    verbose=True,
    backstory="""Skilled in data analysis, you provide valuable insights into market trends and 
    help identify lucrative investment opportunities."""
)

broker = Agent(
    role='Real Estate Broker',
    goal=f'Identify potential properties for investment around {address}',
    verbose=True,
    backstory="""As a seasoned real estate broker, you have access to a wide network of properties 
    and can assist in identifying prime investment opportunities."""
)

# Define tasks
property_analysis = Task(
    description=f"Analyze the investment potential of properties around {address}",
    expected_output="Summary report on investment potential",
    tools=[search_tool, ContentTools().read_content],  
    agent=researcher,
    async_execution=True
)

market_analysis = Task(
    description=f"Conduct a detailed market analysis for {address}",
    expected_output="Market trends and insights report",
    tools=[search_tool, ContentTools().read_content],  
    agent=analyst,
    async_execution=True
)

property_search = Task(
    description=f"Identify potential properties for investment around {address}",
    expected_output="List of potential properties with investment analysis",
    tools=[search_tool, ContentTools().read_content],  
    agent=broker,
    async_execution=True
)

manager_task = Task(
    description=f"""Oversee the integration of research findings, market analysis, and property search 
    to develop comprehensive investment strategies for properties around {address}. Ensure accuracy 
    and feasibility of the proposed strategies.""",
    expected_output=f'Comprehensive investment strategies for properties around {address}.',
    agent=manager
)

# Form crew
crew = Crew(
    agents=[manager, researcher, analyst, broker],
    tasks=[property_analysis, market_analysis, property_search, manager_task],
    process=Process.hierarchical
)

# Kick off crew's work
if st.sidebar.button("Start Analysis"):
    st.sidebar.text("Analysis in progress...")
    results = crew.kickoff()
    st.sidebar.success("Analysis completed!")

    # Display results
    st.subheader("Crew Work Results:")
    st.write(results)

# Language model
ollama_llm = Ollama(model="mistral")
