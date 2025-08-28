# Import Libraries
from crewai import Crew, Process
from crewai.project import crew, before_kickoff, after_kickoff
import os
from dotenv import load_dotenv
from custom_base_crew import CustomBaseCrew

class EvaluationCrew(CustomBaseCrew):

    @crew
    def crew(self) -> Crew:
        # Create Crew instance as usual with agents and tasks
        return Crew(
            name="EvaluationCrew",
            agents=[self.evaluation_agent()],
            tasks=[self.evaluation_task()],
            process=Process.sequential,
            verbose=True,
            output_log_file=self.output_log_file
        )