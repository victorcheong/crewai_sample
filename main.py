# Import Libraries
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from tools import pdf_parser_tool, Image2TextTool
from dotenv import load_dotenv
import os
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

@CrewBase
class PDFParsingCrew():

    def __init__(self):
        self.agents_config = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
        self.tasks_config = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')
        self.tools_config = os.path.join(os.path.dirname(__file__), 'config', 'tools.yaml')

    @before_kickoff
    def before_kickoff(self, inputs):
        return inputs

    @after_kickoff
    def after_kickoff(self, result):
        return result

    @agent
    def parse_pdf_agent(self) -> Agent:
        # llm = ChatOllama(model = 'ollama/gemma3:12b-it-qat', temperature = 0)
        agent_instance = Agent(
            config=self.agents_config['parse_pdf_agent'],
            tools=[pdf_parser_tool],
            # llm=llm,
            verbose=True,
            reasoning = False,
            max_reasoning_attempts=0
        )
        return agent_instance

    @agent
    def image_to_text_agent(self) -> Agent:
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        agent_instance = Agent(
            config=self.agents_config['image_to_text_agent'],
            tools=[Image2TextTool(llm)],
            verbose=True,
            reasoning = False,
            max_reasoning_attempts=0
        )
        return agent_instance

    @task
    def parse_pdf_task(self) -> Task:
        return Task(
            config=self.tasks_config['parse_pdf_task']
    )

    @task
    def image_to_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_to_text_task']
        )

    @crew
    def crew(self) -> Crew:
        # Create Crew instance as usual with agents and tasks
        c = Crew(
            name="PDFParsingCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

        return c

if __name__ == "__main__":
    load_dotenv()
    pdf_path = os.getenv("PDF_DOCUMENT_FILE_PATH")
    my_crew = PDFParsingCrew()
    result = my_crew.crew().kickoff(inputs = {"pdf_document_file_path": pdf_path})