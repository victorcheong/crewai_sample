# Import Libraries
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from tools import PDFParserTool, Image2TextTool, ChunkTextTool, VectorizeTextQATool, EvaluationTool
import os
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
import yaml

@CrewBase
class PDFParsingCrew():

    # Agents and Tasks configuration
    agents_config = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
    tasks_config = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')

    @before_kickoff
    def before_kickoff(self, inputs):
        return inputs

    @after_kickoff
    def after_kickoff(self, result):
        return result

    @agent
    def parse_pdf_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['parse_pdf_agent'], 
            tools=[PDFParserTool()],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

    @agent
    def image_to_text_agent(self) -> Agent:
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        return Agent(
            config=self.agents_config['image_to_text_agent'], 
            tools=[Image2TextTool(llm)],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )
    
    @agent
    def chunk_text_agent(self) -> Agent:
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        return Agent(
            config=self.agents_config['chunk_text_agent'], 
            tools=[ChunkTextTool(llm)],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )
    
    @agent
    def vectorize_text_qa_agent(self) -> Agent:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
        return Agent(
            config=self.agents_config['vectorize_text_qa_agent'], 
            tools=[VectorizeTextQATool(embedding_model, llm)],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'], 
            tools=[EvaluationTool()],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

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

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task']
    )

    @crew
    def crew(self) -> Crew:
        # Create Crew instance as usual with agents and tasks
        return Crew(
            name="PDFParsingCrew",
            agents=[self.parse_pdf_agent(), self.image_to_text_agent(), self.chunk_text_agent(), self.vectorize_text_qa_agent()],
            tasks=[self.parse_pdf_task(), self.image_to_text_task(), self.chunk_text_task(), self.vectorize_text_qa_task()],
            process=Process.sequential,
            verbose=True,
        )