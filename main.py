# Import Libraries
from supervisor import Supervisor
from diagram_flow import EvaluationFlow
import asyncio

async def main():
    flow = EvaluationFlow()
    result = await flow.kickoff_async()
    print(result)
    flow.save_dynamic_graph("flow_graph")

if __name__ == "__main__":
    # supervisor = Supervisor()
    # result = supervisor.run(initial_inputs={})

    asyncio.run(main())