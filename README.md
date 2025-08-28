Note:
1. Tasks invoke tools, not agent invoke tools. Important to change tools, tasks and agents.
2. Run python main.py for crew to kickoff.
3. Arbitary objects (such as base64-decoded strings, vector store, etc.) should not be passed from 1 tool to another, as these objects are not serialisable and may get corrupted. Recommend to use these objects within a tool. Alternatively, can save to disk.
4. I have created 2 possible approaches, 1 is synchronous (using supervisor), and the other is async (using Flow, which also produces a diagram as byproduct)

In your .env file, do add the following configs:
OPENAI_API_KEY
PDF_DOCUMENT_FILE_PATH
GOLDEN_QA_DATASET
PDF_PARSER_CREW_OUTPUT_LOG_FILE
EVALUATION_CREW_OUTPUT_LOG_FILE
PLOT_SAVE_DIR
PDF_IMAGES_DIR
SIMILARITY_THRESHOLD = 0.8