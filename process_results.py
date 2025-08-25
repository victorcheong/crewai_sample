from dotenv import load_dotenv
import os
import json
from datetime import datetime

class ProcessResults:
    def __init__(self):
        load_dotenv()
        self.pdf_parser_output_log_file = os.getenv("PDF_PARSER_CREW_OUTPUT_LOG_FILE")
        with open(self.pdf_parser_output_log_file, 'r') as file:
            self.parser_crew_output = json.load(file)

        self.evaluation_output_log_file = os.getenv("EVALUATION_CREW_OUTPUT_LOG_FILE")
        with open(self.evaluation_output_log_file, 'r') as file:
            self.evaluation_crew_output = json.load(file)

    def compute_time(self):
        start = self.parser_crew_output[0]['timestamp']
        start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end = self.evaluation_crew_output[-1]['timestamp']
        end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        duration = end - start
        return duration.total_seconds()
