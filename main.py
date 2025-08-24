# Import Libraries
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from tools import PDFParserTool, Image2TextTool, ChunkTextTool, VectorizeTextQATool
from dotenv import load_dotenv
import os
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings

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
            tools=[PDFParserTool()],
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
    
    @agent
    def chunk_text_agent(self) -> Agent:
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        agent_instance = Agent(
            config=self.agents_config['chunk_text_agent'],
            tools=[ChunkTextTool(llm)],
            verbose=True,
            reasoning = False,
            max_reasoning_attempts=0
        )
        return agent_instance
    
    @agent
    def vectorize_text_qa_agent(self) -> Agent:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        agent_instance = Agent(
            config=self.agents_config['vectorize_text_qa_agent'],
            tools=[VectorizeTextQATool(embedding_model, llm)],
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

    @task
    def chunk_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['chunk_text_task']
        )

    @task
    def vectorize_text_qa_task(self) -> Task:
        return Task(
            config=self.tasks_config['vectorize_text_qa_task']
        )

    @crew
    def crew(self) -> Crew:
        # Create Crew instance as usual with agents and tasks
        return Crew(
            name="PDFParsingCrew",
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

if __name__ == "__main__":
    load_dotenv()
    pdf_path = os.getenv("PDF_DOCUMENT_FILE_PATH")
    input_query = os.getenv("USER_QUERY")
    my_crew = PDFParsingCrew()
    result = my_crew.crew().kickoff(inputs = {"pdf_document_file_path": pdf_path, "user_query": input_query})