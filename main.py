import requests
import cv2
import base64
import io
import sys
import os
import time
import argparse
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import logging
import base64
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers.utils import TRANSFORMERS_CACHE
import numpy as np
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

model_id = 'microsoft/Florence-2-large'
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", trust_remote_code=True)
    model = model.to(device)
model.eval()  # Set the model to evaluation mode
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def generate_object_detection(image):
    task_prompt = '<OD>'

    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    return parsed_answer


# open footage.mp4 
# request each /api/object_detection
# save each frame and add bbox

def frame_to_data_url(frame, image_format='png'):
    # Encode the frame as an image
    success, encoded_image = cv2.imencode(f'.{image_format}', frame)
    
    if not success:
        raise ValueError("Could not encode the frame")
    
    # Convert the encoded image to a base64 string
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    
    # Create the data URL
    data_url = f'data:image/{image_format};base64,{base64_string}'
    
    return data_url


def numpy_image_to_bytes_io(image: np.ndarray, image_format='png') -> BytesIO:
    # Encode the image as an image file
    success, encoded_image = cv2.imencode(f'.{image_format}', image)
    
    if not success:
        raise ValueError("Could not encode the image")
    
    # Convert the encoded image to bytes
    image_bytes = encoded_image.tobytes()
    
    # Create a BytesIO object from the bytes
    image_bytes_io = BytesIO(image_bytes)
    
    return image_bytes_io

def main():
    url = 'http://localhost:5000/api/object_detection'
    files = {'file': open('footage.mp4', 'rb')}
    #make request for each frame
    #split video into frames
    #for each frame
    #send request
    #save frame with bbox
    #display frame

    #get video
    cap = cv2.VideoCapture('footage.mp4')
    #get frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #get frame width
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #get frame height
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #get frame rate
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    #get codec
    codec = cv2.VideoWriter_fourcc(*'XVID')
    #create video writer
    out = cv2.VideoWriter('output.avi', codec, frame_rate, (frame_width, frame_height))
    #loop through frames


    for i in range(frame_count):
        print(f"Processing frame {i}/{frame_count}")
        #read frame
        ret, frame = cap.read()
        image = Image.open(numpy_image_to_bytes_io(frame)).convert("RGB")
        detection_result = generate_object_detection(image)
        #get response
        #get bbox

        bbox = detection_result['bboxes']
        #draw bbox
        for i in range(len(bbox)):
            x1, y1, x2, y2 = map(int, bbox[i])  # Convert coordinates to integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # write the class name
            cv2.putText(frame, detection_result["labels"][i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #save frame with bbox
        out.write(frame)
        #display frame
        #wait for key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #release video
    cap.release()
    out.release()

main()