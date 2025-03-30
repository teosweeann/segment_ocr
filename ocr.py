import numpy as np
import cv2
import os
import time
import io, re
from PIL import Image
from pytesseract import image_to_string
import base64
import requests

API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY")

import subprocess

def ping_google():
    # Ping www.google.com once
    output = subprocess.run(
        ["ping", "-c", "1", '-W', '1', "www.google.com"],
        capture_output=True, text=True)
    # Check if the ping command was successful
    if output.returncode == 0: return True
    return False

def tesseract(image, lang=None):
    if image is None:
        print("ERROR: no image file found")
        exit()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6 preserve_interword_spaces=1'
    # Run OCR on the image
    if lang=='chi':
        text = image_to_string(gray_image, config=custom_config, lang='chi_sim')
    else:
        text = image_to_string(gray_image, config=custom_config, lang='eng')
    return text

def image_to_text(image):
    tt = _transcribe(image)
    tt = tt.replace("\\text{", "\\mathrm{")
    tt = tt.replace('\\( ','$$').replace(" \\)", '$$')
    tt = tt.replace('\\(','$$').replace("\\)", '$$')
    tt = tt.replace('\\[ ','$$').replace(" \\]", '$$')
    tt = tt.replace('\\[','$$').replace("\\]", '$$')
    tt = re.sub(r"\\textit{(.+?)}", r"*\1*", tt)
    tt = re.sub(r"\\textbf{(.+?)}", r"**\1**", tt)
    tt = re.sub(r"\\cite{(.+?)}", r"[\1]", tt)
    tt = re.sub(r"\\hfill", r" ", tt)
    tt = re.sub(r"\\hfill", r" ", tt)
    tt = re.sub(r"\$\$[\r\n\s]*\\hspace\{[\d\.]+(?:cm|pt)\}\s*\((\d+)\)", r"\\tag{\1}$$", tt)
    #tt = re.sub(r"\S\$\$\s*(?:\\qquad|\s{4}|\s{2})*\s*(?:\\quad|\s{2})*\s*\((\d+)\)", r"\\tag{\1}$$", tt)
    tt = re.sub(r"\S\$\$\s*(?:\\qquad|\s{4}|\s{2}|\s{1})*\s*(?:\\quad|s{1})*\s*\((\d+)\)$$", r"\\tag{\1}$$", tt)
    tt = re.sub(r"\$\$[\r\n\s]+\((\d+)\)", r"\\tag{\1}$$", tt)
    tt = re.sub(r"\n\$\$(\S+[^\$]*\S+)\$\$\n", "\n$$\n\\1\n$$\n", tt)
    return tt

def make_md_1(image, model='gpt'):
    nol, l_height = estimate_text_lines(image)
    if nol > 20:
        im1, im2 = segment_image(image)
        return make_md_1(im1, model=model) + '\n' + make_md_1(im2, model=model)
    else: 
        if model=='gpt':
            return extract_text_from_image(image)
        else:
            return extract_text_from_image_llama(image)

def _transcribe(image):
    nol, l_height = estimate_text_lines(image)
    if nol > 20:
        img1, img2 = segment_image(image)
        if not img1 is None:
            return _transcribe(img1)  + '\n' + _transcribe(img2)
    output = ask( 
            'Convert this image containing text to markdown text with latex equation support. '
            'Include partial sentences at the start or end.  Remove any unnecessary line breaks. '
            'Try your best to not exclude any content from the image. '
            'Return only the output text with no explanation text. Omit wrapper text. ',
            image
            )
    return output

def classify_image(image):
    output = ask( 
            'Please help me classify this image as ["text only", "figure", "table", "flowchart", '
            '"caption", "figure containing text", "separate figure and text", "figure with caption", '
            '"table with caption"]. Omit wrapper text and provide the answer only. ',
            image ).lower().strip('"').strip()
    return output

def table_cap_position(image):
    output = ask( 
            "Locate the tables in this page. Do the captions of these tables preceed or follow the tables? "
            "These captions start with 'Table'. "
            "Omit wrapper text and reply with 'preceed' or 'follow', and also give the locations. "
            "Check three times before replying. ",
            image ).lower().strip("'").strip()
    if 'follow' in output: return -1
    return 1

def figure_cap_position(image):
    output = ask( 
            "Locate the figures in this page. Does the caption of these figures preceed or follow these figures? "
            "Omit wrapper text and reply with 'preceed' or 'follow', and also give the locations. "
            "Check three times before replying. ",
            image ).lower().strip("'").strip()
    if 'follow' in output: return -1
    return 1

def estimate_text_lines(img, display=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    proj = np.sum(binary // 255, axis=1)
    w = binary.shape[1]
    proj_threshold = 0.01 * w
    text_rows = proj > proj_threshold
    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(text_rows):
        if val and not in_line:
            in_line = True
            start = i
        elif not val and in_line:
            in_line = False
            end = i - 1
            lines.append((start, end))
    if in_line:
        lines.append((start, len(text_rows) - 1))

    num_lines = len(lines)
    line_heights = [end - start + 1 for start, end in lines]
    avg_line_height = np.mean(line_heights) if line_heights else 0
    return num_lines, avg_line_height

def find_segmentation_row(binary):
    H = binary.shape[0]
    center = H // 2
    row_sum = np.sum(binary, axis=1) / 255
    candidate_row = None
    min_text = np.inf
    for y in range(center//3):
        if row_sum[center+y] < min_text:
            min_text = row_sum[center+y]
            candidate_row = center+y
        if min_text == 0:
            break
        if row_sum[center-y] < min_text:
            min_text = row_sum[center-y]
            candidate_row = center-y
        if min_text == 0:
            break
    return candidate_row

def segment_image(image, search_range=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    seg_row = find_segmentation_row(binary)
    if seg_row is None:
        print("Could not determine a segmentation line.")
        return None, None
    print(f"Segmentation row chosen: {seg_row}")
    top_part = image[:seg_row, :]
    bottom_part = image[seg_row:, :]
    return top_part, bottom_part


if ping_google():
    from ai_wrapper import ask_gpt as ask
else:
    from ai_wrapper import ask_gemma3 as ask
