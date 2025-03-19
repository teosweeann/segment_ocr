import ollama
import os
import time
from PIL import Image
import cv2
import numpy as np
import io, re
import base64
import requests

API_KEY = os.environ.get("OPENAI_API_KEY")

def encode_image_to_base64(img, fmt="JPEG"):
    if isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    else:
        image = img.copy()
    if image.mode == "RGBA":
        image = image.convert("RGB")  # Convert to RGB mode
    buffered = io.BytesIO()
    image.save(buffered, format=fmt)
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_to_tmpfile(img, fmt="JPEG", fix_color_swap=False):
    import tempfile
    if isinstance(img, np.ndarray):
        image = Image.fromarray(img)
    else:
        image = img.copy()
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Optionally swap red and blue channels if colors appear swapped
    if fix_color_swap:
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
    
    # Save to a temporary JPEG file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file, format="JPEG")
        return temp_file.name

def ask_gpt(question, image):
    if not image is None:
        #image_path = save_image_to_tmpfile(image)
        base64_image = encode_image_to_base64(image)
        msg = { "model": "gpt-4o", "messages": [{
             "role":"user", 'content': question },{
             "role": "user", "content": [{ 
                 "type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{base64_image}"},},],}] }
    else:
        msg = { "model": "gpt-4o", "messages": [{
             "role":"user", 'content': question }] }
    for i in range(5):
        try:
            output = requests.post("https://api.openai.com/v1/chat/completions",
                headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json = msg).json()
            output = output['choices'][0]['message']['content']
            if 'sorry' in output and "can't" in output or 'unable to process' in output:
                cv2.imwrite("debug.png", image)
            else: 
                break
        except:
            raise
            time.sleep(2*(i+1))
    output = output.replace('```markdown','')
    output = output.replace('```plaintext','')
    output = output.replace('```','').strip('"').strip()
    return output

def ask_gemma3(question, image, temp=0.7):
    if not image is None:
        image_path= image_to_tmpfile(image)
        msg = [ {
            "role": "system",
            "content": question }, {
            "role": "user", "content": '', "images": [image_path] }]
    else:
        msg = [ {
            "role": "system",
            "content": question } ]
    print(msg)
    output = ollama.chat(
        model='gemma3:27b', 
        messages = msg,
        options = {'temperature': temp}
    )['message']['content'].strip()

    output = output.replace('```markdown','')
    output = output.replace('```plaintext','')
    output = output.replace('```','').strip('"').strip()
    return output
