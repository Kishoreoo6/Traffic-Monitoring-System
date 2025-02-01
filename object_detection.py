import cv2
import os
from ultralytics import YOLO


model = YOLO(r'C:\Users\mural\OneDrive\ドキュメント\mini proj\mini proj-yolov8\yolov8m.pt')


classes = model.names

def process_frame(img):
    
    results = model(img) 
    
    vehicle_count = 0  

    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection[:6]

            label = classes[int(class_id)]
            confidence = round(score, 2)

            if label in ['car', 'truck', 'bus', 'motorbike']:
                vehicle_count += 1

              
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {confidence}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

   
    if vehicle_count < 5:
        density_text = "TRAFFIC DENSITY: LOW"
    elif 5 <= vehicle_count <= 15:
        density_text = "TRAFFIC DENSITY: MODERATE"
    else:
        density_text = "TRAFFIC DENSITY: HIGH"

   
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3
    text_color = (255, 255, 255) 
    bg_color = (0, 0, 255) 

  
    (text_width, text_height), baseline = cv2.getTextSize(density_text, font, font_scale, font_thickness)

    
    text_offset_x = 10
    text_offset_y = 40
    box_coords = ((text_offset_x - 5, text_offset_y + 5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)

    
    cv2.putText(img, density_text, (text_offset_x, text_offset_y), font, font_scale, text_color, font_thickness)

    return img


def process_input(input_path):
    try:
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(input_path)
            processed_img = process_frame(img)
            output_image_path = os.path.splitext(input_path)[0] + '_output.jpg'
            cv2.imwrite(output_image_path, processed_img)
            print(f"Output image saved at: {output_image_path}")

        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(input_path)
            output_video_path = os.path.splitext(input_path)[0] + '_output.avi'

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

          
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 10 == 0:  # Log progress
                    print(f"Processed {frame_count} frames...")

            cap.release()
            out.release()

            if os.path.exists(output_video_path):
                print(f"Output video successfully saved at: {output_video_path}")
            else:
                print("Error saving output video.")
        else:
            print("Unsupported file format. Please provide an image (.jpg, .jpeg, .png) or video (.mp4, .avi, .mov, .mkv).")

    except Exception as e:
        print(f"Error in processing input: {e}")

# Example usage
input_file_path = r'C:\Users\mural\OneDrive\ドキュメント\mini proj\mini proj-yolov8\input 2 comp.png'
process_input(input_file_path)
