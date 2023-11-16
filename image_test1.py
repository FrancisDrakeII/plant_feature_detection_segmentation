import cv2
from ultralytics import YOLO
import subprocess
import smtplib
from email.message import EmailMessage

# Load model and image
path = "IMG_8150.JPG"
model_name = "AI-8-TS.pt"
model = YOLO(model_name)
results = model(path)

subprocess.run(["yolo", "mode=predict", "segment", "model=AI-8-TS.pt", "source=IMG_8428.JPG", "save=True", "conf=0.90", "boxes=False"])

"""
# Load image using OpenCV
image = cv2.imread(path)

# Iterate over detected objects and draw boxes
for obj in results[0]:
    # Convert Ultralytics box format to OpenCV box format (x_min, y_min, x_max, y_max)
    box = obj.boxes.xyxy.cpu().numpy().astype(int)
    # print(box)
    # Draw the box on the image
    cv2.rectangle(image, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 255, 0), 2)

# Save the result
cv2.imwrite("result_" + path, image)
"""

def send_email(subject, body, to_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'zihuanteng@gmail.com'  # replace with your email
    msg['To'] = to_email

    # Gmail settings
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('zihuanteng@gmail.com', 'ubbk xebd faqb ckif')  # replace with your email and password
    server.send_message(msg)
    server.quit()

send_email('Test', 'Run successful', 'zihuanteng@gmail.com')