from crewai.tools import BaseTool
import fitz
import base64
from typing import Any
from langchain_core.messages import HumanMessage
import time
import re

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

    def _run(self, pdf_document_file_path: str) -> dict:
        doc = fitz.open(pdf_document_file_path)
        images_paths = []
        for i, page in enumerate(doc):
            img_bytes = pdf_page_to_png_bytes(page)
            img_b64 = image_bytes_to_base64(img_bytes)
            image_path = f"debug_page_{i+1}.png"
            save_b64_image(img_b64, image_path)
            images_paths.append(image_path)
        return {"image_paths": images_paths}

pdf_parser_tool = PDFParserTool() #create the tool so that main.py can import it

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

    def _run(self, image_paths: list) -> list:
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

        # Typically, parse and return structured output
        return all_extracted_texts