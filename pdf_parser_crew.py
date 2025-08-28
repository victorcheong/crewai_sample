# Import Libraries
from crewai import Crew, Process
from crewai.project import crew
import os
from custom_base_crew import CustomBaseCrew

class PDFParsingCrew(CustomBaseCrew):

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