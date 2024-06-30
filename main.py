import requests
import cv2
import base64
import io


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
        #send request
        encoded_string = frame_to_data_url(frame)
        print(encoded_string[:10])
        response = requests.post("http://127.0.0.1:5000/api/object_detection", json={"image": encoded_string})
        #get response
        data = response.json()
        #get bbox
        print(data)
        bbox = data["detection"]['bboxes']
        #draw bbox
        for box in bbox:
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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