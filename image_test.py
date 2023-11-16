import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('AI-8-9-TS.pt')

path = "IMG_8150.JPG"
cap = cv2.VideoCapture(path)

# Create a text object
font = cv2.FONT_HERSHEY_SIMPLEX

# Set the font scale
fontScale = 1

# Set the thickness
thickness = 2

# Set the color
color = (0, 0, 0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        # mask = (results[0].boxes.data[:, -2]) > 0.9
        # print(results[0].boxes.data)
        tensor_data = results[0].boxes.data
        confidence_threshold = 0.75
        high_conf_detections = tensor_data[tensor_data[:, 4] > confidence_threshold]

        # print(results[0].boxes)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # comparison_image = cv2.hconcat([frame, annotated_frame])
        annotated_frame = results[0].plot()
        comparison_image = cv2.hconcat([frame, annotated_frame])

        # bud, pod, white, yellow = 0, 0, 0, 0
        # for box in result.boxes:
        #     if box.cls[0].item() == 3:
        #         yellow += 1
        #         print(f"yellow {yellow} {box.xyxy[0].tolist()}")
        #     elif box.cls[0].item() == 2:
        #         white += 1
        #         print(f"white {white} {box.xyxy[0].tolist()}")
        #     elif box.cls[0].item() == 1:
        #         pod += 1
        #         print(f"white {pod} {box.xyxy[0].tolist()}")
        #     else:
        #         bud += 1
        #         print(f"white {bud} {box.xyxy[0].tolist()}")
        #
        # print(f"white : {white}, yellow: {yellow}, bud: {bud}, pod: {pod}")



        # cv2.putText(annotated_frame, "white: " + str(white), (10, 30), font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.putText(annotated_frame, "yellow: " + str(yellow), (10, 60), font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.putText(annotated_frame, "bud: " + str(bud), (10, 90), font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.putText(annotated_frame, "pod: " + str(pod), (10, 120), font, fontScale, color, thickness, cv2.LINE_AA)

        # Display the annotated frame
        # cv2.imshow("Carinata Model Detection", annotated_frame)
        cv2.imwrite("result_" + path, comparison_image)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
