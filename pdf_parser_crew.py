# Import Libraries
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from tools import PDFParserTool, Image2TextTool, ChunkTextTool, VectorizeTextQATool, EvaluationTool
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

@CrewBase
class PDFParsingCrew():
    load_dotenv()
    # Agents and Tasks configuration
    agents_config = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
    tasks_config = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')
    output_log_file = os.getenv("PDF_PARSER_CREW_OUTPUT_LOG_FILE")

    def __init__(self, llm, vision_llm):
        self.llm = llm
        self.vision_llm = vision_llm

    @before_kickoff
    def before_kickoff(self, inputs):
        if self.output_log_file:
            open(self.output_log_file, 'w').close()
        return inputs

    @after_kickoff
    def after_kickoff(self, result):
        return result

    @agent
    def parse_pdf_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['parse_pdf_agent'], 
            tools=[PDFParserTool(result_as_answer=True)],
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

    @agent
    def image_to_text_agent(self) -> Agent:
        try:
            llm = ChatOpenAI(model=self.vision_llm, temperature=0)
        except Exception as e:
            llm = ChatOllama(model=self.vision_llm, temperature=0)
        return Agent(
            config=self.agents_config['image_to_text_agent'], 
            tools=[Image2TextTool(llm)],
            result_as_answer=True,
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )
    
    @agent
    def chunk_text_agent(self) -> Agent:
        try:
            llm = ChatOpenAI(model=self.llm, temperature=0)
        except Exception as e:
            llm = ChatOllama(model=self.llm, temperature=0)
        return Agent(
            config=self.agents_config['chunk_text_agent'], 
            tools=[ChunkTextTool(llm)],
            result_as_answer=True,
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )
    
    @agent
    def vectorize_text_qa_agent(self) -> Agent:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            llm = ChatOpenAI(model=self.llm, temperature=0)
        except Exception as e:
            llm = ChatOllama(model=self.llm, temperature=0)
        return Agent(
            config=self.agents_config['vectorize_text_qa_agent'], 
            tools=[VectorizeTextQATool(embedding_model, llm)],
            result_as_answer=True,
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'], 
            tools=[EvaluationTool()],
            result_as_answer=True,
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
            output_log_file=self.output_log_file
        )