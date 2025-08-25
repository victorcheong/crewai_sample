Note:
1. Tasks invoke tools, not agent invoke tools. Important to change tools, tasks and agents.
2. Run python main.py for crew to kickoff.
3. Arbitary objects (such as base64-decoded strings, vector store, etc.) should not be passed from 1 tool to another, as these objects are not serialisable and may get corrupted. Recommend to use these objects within a tool. Alternatively, can save to disk.
4. I have created 2 possible approaches, 1 is synchronous (using supervisor), and the other is async (using Flow, which also produces a diagram as byproduct)