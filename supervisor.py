from custom_base_crew import CustomBaseCrew
from evaluation_crew import EvaluationCrew
from pdf_parser_crew import PDFParsingCrew
import json
from dotenv import load_dotenv
import os

class Supervisor():

    def __init__(self, llm, vision_llm, embedding_model, question, ground_truth):
        load_dotenv()
        self.pdf_path = os.getenv("PDF_DOCUMENT_FILE_PATH")
        self.input_query = question
        self.base_crew = CustomBaseCrew(llm, vision_llm, embedding_model, ground_truth)

    def run(self, initial_inputs):
        first_crew = self.base_crew.create_pdf_parsing_crew(PDFParsingCrew)
        first_crew.set_output_log_file(os.getenv("PDF_PARSER_CREW_OUTPUT_LOG_FILE"))
        second_crew = self.base_crew.create_evaluation_crew(EvaluationCrew)
        second_crew.set_output_log_file(os.getenv("EVALUATION_CREW_OUTPUT_LOG_FILE"))
        
        answer = first_crew.crew().kickoff(inputs = {"pdf_document_file_path": self.pdf_path, "user_query": self.input_query})
        with open("retrieved_docs.json", "r") as f:
            json_str = f.read()
            retrieved_docs = json.loads(json_str)

        second_result = second_crew.crew().kickoff(inputs={"answer": answer.raw, "retrieved_docs": retrieved_docs, "user_query": self.input_query})

        # Optionally combine or return results as needed
        return {
            "first_crew_result": answer,
            "second_crew_result": second_result
        }