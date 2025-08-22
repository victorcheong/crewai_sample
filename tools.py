from crewai.tools import BaseTool
import fitz
import base64

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

class PDFParserTool(BaseTool):
    name: str = "pdf_parser_tool"
    description: str = "Converts PDF pages to base64 encoded images."

    def _run(self, pdf_document_file_path: str) -> list:
        doc = fitz.open(pdf_document_file_path)
        all_img_strs = []
        for page in doc:
            img_bytes = pdf_page_to_png_bytes(page)
            img_b64 = image_bytes_to_base64(img_bytes)
            all_img_strs.append(img_b64)
        return all_img_strs

pdf_parser_tool = PDFParserTool() #create the tool so that main.py can import it