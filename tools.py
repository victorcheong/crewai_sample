from crewai.tools import BaseTool
import fitz
import base64
from typing import Any
from langchain_core.messages import HumanMessage
import time
from langchain_community.vectorstores import FAISS

def pdf_page_to_png_bytes(page):
    """
    Converts a pdf page to an image.

    :param page: The PDF page to convert.
    :return: The image of the PDF page in bytes.
    """
    pix = page.get_pixmap()
    return pix.tobytes(output="png")

def image_bytes_to_base64(img_bytes):
    """
    Converts image bytes to a base64 encoded string.

    :param img_bytes: The image bytes to convert.
    :return: The base64 encoded string of the image.
    """
    return base64.b64encode(img_bytes).decode("utf-8")

def save_b64_image(b64_string, filename):
    try:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64_string))
        print(f"{filename} saved successfully.")
    except Exception as e:
        print(f"Failed to save image: {e}")

class PDFParserTool(BaseTool):
    name: str = "pdf_parser_tool"
    description: str = "Converts PDF pages to base64 encoded images."

    def _run(self, pdf_document_file_path: str) -> list:
        doc = fitz.open(pdf_document_file_path)
        images_paths = []
        for i, page in enumerate(doc):
            img_bytes = pdf_page_to_png_bytes(page)
            img_b64 = image_bytes_to_base64(img_bytes)
            image_path = f"debug_page_{i+1}.png"
            save_b64_image(img_b64, image_path)
            images_paths.append(image_path)
        return {"image_paths": images_paths}

def prompt_func(data):
    text = data['text']
    image = data['image']

    image_part = {
        "type": 'image_url',
        "image_url": { "url": f"data:image/png;base64,{image}"},
    }

    content_parts =[]

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content = content_parts)]

class Image2TextTool(BaseTool):
    name: str = "Image2TextTool"
    description: str = "Extracts text and tables from images."
    llm: Any

    def __init__(self, llm_client):
        super().__init__(llm=llm_client)
        self.llm = llm_client

    def _run(self, image_paths: list) -> str:
        all_extracted_texts = []
        for i, image_path in enumerate(image_paths):
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            image_string = image_bytes_to_base64(image_bytes)
            # Construct prompt or call LLM directly with the image string data for OCR/text extraction
            messages = prompt_func({"text": """
            Extract all texts, tables, and images from this PDF page.

            For images:

            Provide a detailed, but yet concise description.

            Mention the main subjects, notable objects, actions, colors, setting, style, and any visible text.

            Avoid generic or overly brief summaries.

            For text and tables:

            Extract and transcribe content fully and clearly.
                                    
            If you see tables / images such as the following:
            1. Task X.................... Yes as 2, No as 3
            What it means is that you have to think of it like a decision tree. so this means if Task X is "Yes", go to question 2 of the table.
            Else, you go to question 3 of the table.     
            For such tables / images, please translate the texts to a decision tree, instead of just listing down the table as it is in the image.
            Do also construct an inverted decision tree, so that you can reverse engineer and identify the first task that needs to be done, 
            based on the status of the last task.          
            
            """, "image": image_string})
            # Call the LLM with prompt to get text output (adjust this call to your LLM client's API)
            raw_response = self.llm.invoke(messages)
            all_extracted_texts.append(raw_response.content)
            time.sleep(10)

        all_extracted_texts = "\n".join(all_extracted_texts)
        # Typically, parse and return structured output
        return {"extracted_text": all_extracted_texts}
    
class ChunkTextTool(BaseTool):
    name: str = "ChunkTextTool"
    description: str = "Chunks text into smaller segments."
    llm: Any

    def __init__(self, llm_client):
        super().__init__(llm=llm_client)
        self.llm = llm_client

    def _run(self, extracted_text: str) -> list:
        # Prepare a prompt giving the LLM the mini-chunks and asking it to group them.
        prompt = f"""
        You will receive a concatenated string of extracted texts from a larger document. 
        Your task is to freely split the concatenated string into mini chunks to form the most coherent, semantically meaningful, and 
        logically organized chunks possible.
        Feel free to recreate chunks by combining, splitting, or reordering texts based on their meaning and thematic consistency.
        Please output the resulting chunks in a clear, numbered format. 
        Example format:
        Chunk 1 Content: ...

        Chunk 2 Content: ...

        Concatenated string:
        {extracted_text}
        """
        # Get grouping suggestions from the LLM
        response = self.llm.invoke(prompt)
        # (Optionally, parse out the chunk groupings here)

        return {"generated_chunks": response.content}

class VectorizeTextQATool(BaseTool):
    name: str = "VectorizeTextQATool"
    description: str = "Vectorizes text into embeddings and returns answer based on user query"
    embedding_llm: Any
    llm: Any

    def __init__(self, embedding_llm_client, llm_client):
        super().__init__(embedding_llm=embedding_llm_client, llm=llm_client)
        self.embedding_llm = embedding_llm_client
        self.llm = llm_client

    def _run(self, chunks: list, user_query: str) -> str:
        # Creates vector store and encodes texts internally
        vectorstore = FAISS.from_texts(chunks, self.embedding_llm)
        retrieved_docs = vectorstore.similarity_search(user_query, k=3)
        prompt = f"""You are a document analysis assistant. Your only source of information is the provided context.
  
        STRICT GUIDELINES:
        1. Use ONLY the information from the context based on the meaning and facts expressed, not limited to exact wording.
        2. Do NOT use any external information or make assumptions beyond the context.
        3. If the answer cannot be derived from the provided context, reply: "This information is not available in the provided context."
        4. Provide clear and factual answers without unnecessary elaboration.
        5. Answer concisely and only what is asked.
        
        Instructions:
        Use the context below to answer the question, interpreting the information by understanding its content, even if exact phrases differ.
        
        Question: {user_query}
        Context:
        {retrieved_docs}"""
        response = self.llm.invoke(prompt)
        return {"answer": response.content}