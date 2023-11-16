import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('AI-8-9-X.pt')

# Path to the directory containing the images
input_dir = "AI-9 PHY490 TS"

# Path to the directory where the results will be saved
output_dir = input_dir + " results"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of all the .jpg files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.JPG')]

# Set the font, font scale, thickness, and color for text annotation
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
color = (0, 0, 0)

# Loop through all the .jpg files in the input directory
for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_dir, image_file)
    frame = cv2.imread(image_path)

    # Ensure the image loaded correctly
    if frame is not None:
        # Run YOLOv8 inference on the image
        results = model(frame)
        # print(results[0].boxes.conf)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Optional: Display the annotated frame
        # cv2.imshow("Carinata Model Detection", annotated_frame)
        comparison_image = cv2.hconcat([frame, annotated_frame])

        # Save the annotated frame to the output directory
        result_path = os.path.join(output_dir, "result_" + image_file)
        cv2.imwrite(result_path, comparison_image)

        # Optional: Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        print(f"Failed to load {image_path}")

# Optional: Close the display window
# cv2.destroyAllWindows()
