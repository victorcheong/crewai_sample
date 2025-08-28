# Import Libraries
import os
from supervisor import Supervisor
from process_results import ProcessResults
from plot_results import PlotResults
from diagram_flow import EvaluationFlow
import asyncio
from itertools import product
from dotenv import load_dotenv
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings

async def main():
    flow = EvaluationFlow()
    await flow.kickoff_async()
    flow.save_dynamic_graph("flow_graph")

if __name__ == "__main__":
    load_dotenv()
    llms = ['gemma3:12b-it-qat', 'gpt-4o-mini']
    vision_llms = ['llava:7b', 'gpt-4.1-mini']
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    permutations = list(product(llms, vision_llms))
    golden_dataset = pd.read_excel(os.getenv("GOLDEN_QA_DATASET"))
    for index, row in golden_dataset.iterrows():
        question = row['Question']
        ground_truth = row['Expected Output']
        print(f"Processing Question: {question}")
        scores = []
        times = []
        for permutation in permutations:
            llm, vision_llm = permutation
            print(f"Running with LLM: {llm}, Vision LLM: {vision_llm}")
            supervisor = Supervisor(llm=llm, vision_llm=vision_llm, embedding_model=embedding_model, question=question, \
                                    ground_truth=ground_truth)
            result = supervisor.run(initial_inputs={})
            result = result['second_crew_result'].raw
            score = float(result)
            scores.append(score)
            process_results_obj = ProcessResults()
            duration = process_results_obj.compute_time()
            times.append(duration)
            print(f"Processing time: {duration} seconds")
        plot_results = PlotResults(scores, times, permutations, question)
        plot_results.plot_results()
        break
    # asyncio.run(main())