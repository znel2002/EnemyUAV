import requests
import cv2
import base64
import io


# open footage.mp4 
# request each /api/object_detection
# save each frame and add bbox

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string

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
        #read frame
        ret, frame = cap.read()
        #save frame
        cv2.imwrite('frame.jpg', frame)
        #send request
        encoded_string = encode_image(frame)
        response = requests.post("http://127.0.0.1:5000/api/object_detection", json={"image": encoded_string})
        #get response
        data = response.json()
        #get bbox
        bbox = data["detection"]['bbox']
        #draw bbox
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #save frame with bbox
        out.write(frame)
        #display frame
        cv2.imshow('frame', frame)
        #wait for key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
