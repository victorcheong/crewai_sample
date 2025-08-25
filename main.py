# Import Libraries
from supervisor import Supervisor
from process_results import ProcessResults
from plot_results import PlotResults
from diagram_flow import EvaluationFlow
import asyncio
from itertools import product

async def main():
    flow = EvaluationFlow()
    await flow.kickoff_async()
    flow.save_dynamic_graph("flow_graph")

if __name__ == "__main__":
    llms = ['gpt-4o-mini', 'gpt-4.1-mini']
    vision_llms = ['gpt-4o-mini', 'gpt-4.1-mini']
    permutations = list(product(llms, vision_llms))
    scores = []
    times = []
    for permutation in permutations:
        llm, vision_llm = permutation
        print(f"Running with LLM: {llm}, Vision LLM: {vision_llm}")
        supervisor = Supervisor(llm=llm, vision_llm=vision_llm)
        result = supervisor.run(initial_inputs={})
        result = result['second_crew_result'].raw
        score = float(result)
        scores.append(score)
        process_results_obj = ProcessResults()
        duration = process_results_obj.compute_time()
        times.append(duration)
        print(f"Processing time: {duration} seconds")
    # asyncio.run(main())
    plot_results = PlotResults(scores, times, permutations)
    plot_results.plot_results()