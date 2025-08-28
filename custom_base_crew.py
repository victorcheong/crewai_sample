import os
from crewai.project import CrewBase, agent, task, before_kickoff, after_kickoff
from crewai import Agent, Task
from tools import PDFParserTool, Image2TextTool, ChunkTextTool, VectorizeTextQATool, EvaluationTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

@CrewBase
class CustomBaseCrew:

    agents_config = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
    tasks_config = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')

    def __init__(self, llm, vision_llm, embedding_model, ground_truth):
        self.llm = llm
        self.vision_llm = vision_llm
        self.embedding_model = embedding_model
        self.ground_truth = ground_truth

    def set_output_log_file(self, filepath):
        self.output_log_file = filepath

    @before_kickoff
    def before_kickoff(self, inputs):
        if self.output_log_file:
            open(self.output_log_file, 'w').close()
        return inputs

    @after_kickoff
    def after_kickoff(self, result):
        return result

    def create_pdf_parsing_crew(self, pdf_parsing_crew_class):
        return pdf_parsing_crew_class(self.llm, self.vision_llm, self.embedding_model, self.ground_truth)

    def create_evaluation_crew(self, evaluation_crew_class):
        return evaluation_crew_class(self.llm, self.vision_llm, self.embedding_model, self.ground_truth)

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
        if ':' not in self.vision_llm:
            llm = ChatOpenAI(model=self.vision_llm, temperature=0)
        else:
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
        load_dotenv()
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
        return Agent(
            config=self.agents_config['chunk_text_agent'],
            tools=[ChunkTextTool(self.embedding_model, similarity_threshold)],
            result_as_answer=True,
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )
    
    @agent
    def vectorize_text_qa_agent(self) -> Agent:
        if ':' not in self.llm:
            llm = ChatOpenAI(model=self.llm, temperature=0)
        else:
            llm = ChatOllama(model=self.llm, temperature=0)
        return Agent(
            config=self.agents_config['vectorize_text_qa_agent'], 
            tools=[VectorizeTextQATool(self.embedding_model, llm)],
            result_as_answer=True,
            verbose=True,
            reasoning=False,
            max_reasoning_attempts=0
    )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'], 
            tools=[EvaluationTool(self.ground_truth)],
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