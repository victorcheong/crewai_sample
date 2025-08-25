import asyncio
from pydantic import BaseModel, Field
from typing import List, Tuple

from crewai.flow.flow import Flow, listen, start
from pdf_parser_crew import PDFParsingCrew
from evaluation_crew import EvaluationCrew
from dotenv import load_dotenv
import os
import json
from graphviz import Digraph

load_dotenv()

class EvaluationFlowState(BaseModel):
    pdf_path: str = Field(default_factory=lambda: os.getenv("PDF_DOCUMENT_FILE_PATH"))
    input_query: str = Field(default_factory=lambda: os.getenv("USER_QUERY"))
    answer: str = ""
    retrieved_docs: List = []
    transitions: List[Tuple[str, str]] = []
    final_result: str = ""

class EvaluationFlow(Flow[EvaluationFlowState]):

    def before_kickoff(self, inputs):
        # Initialize transitions list on new run
        self.state.transitions = []
        return inputs

    @start()
    async def parse_pdf(self):
        crew = PDFParsingCrew()
        pdf_result = await crew.crew().kickoff_async(
            inputs={"pdf_document_file_path": self.state.pdf_path, "user_query": self.state.input_query}
        )
        self.state.transitions.append(("start", "parse_pdf"))
        self.state.answer = pdf_result.raw
        return {"answer": self.state.answer}

    @listen("parse_pdf")
    async def load_retrieved_docs(self, answer):
        self.state.transitions.append(("parse_pdf", "load_retrieved_docs"))
        with open("retrieved_docs.json", "r") as f:
            retrieved_docs = json.load(f)
        self.state.retrieved_docs = retrieved_docs
        return retrieved_docs

    @listen("load_retrieved_docs")
    async def evaluate(self, retrieved_docs):
        self.state.transitions.append(("load_retrieved_docs", "evaluate"))
        crew = EvaluationCrew()
        eval_result = await crew.crew().kickoff_async(
            inputs={
                "answer": self.state.answer,
                "retrieved_docs": retrieved_docs,
                "user_query": self.state.input_query,
            }
        )
        self.state.final_result = eval_result.raw
        return self.state.final_result

    def save_dynamic_graph(self, filename):
        dot = Digraph()
        nodes = set()
        for start, end in self.state.transitions:
            nodes.add(start)
            nodes.add(end)
        for node in nodes:
            dot.node(node)
        for start, end in self.state.transitions:
            dot.edge(start, end)
        dot.render(filename, format="png", cleanup=True)