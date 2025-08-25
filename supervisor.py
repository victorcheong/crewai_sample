from evaluation_crew import EvaluationCrew
from pdf_parser_crew import PDFParsingCrew
import json
from dotenv import load_dotenv
import os

class Supervisor():

    def __init__(self, llm, vision_llm):
        load_dotenv()
        self.pdf_path = os.getenv("PDF_DOCUMENT_FILE_PATH")
        self.input_query = os.getenv("USER_QUERY")
        self.first_crew = PDFParsingCrew(llm, vision_llm)
        self.second_crew = EvaluationCrew()

    def run(self, initial_inputs):
        answer = self.first_crew.crew().kickoff(inputs = {"pdf_document_file_path": self.pdf_path, "user_query": self.input_query})
        with open("retrieved_docs.json", "r") as f:
            json_str = f.read()
            retrieved_docs = json.loads(json_str)

        second_result = self.second_crew.crew().kickoff(inputs={"answer": answer.raw, "retrieved_docs": retrieved_docs, "user_query": self.input_query})

        # Optionally combine or return results as needed
        return {
            "first_crew_result": answer,
            "second_crew_result": second_result
        }