from ultralytics import YOLO
from roboflow import Roboflow
import numpy as np
import cv2 as cv
import random
import os

# Load the image
image_path = "test01.jpg"

# Load the YOLO model
model_path = "best.pt"
model = YOLO(model_path)

def predict_image(image_path, expected_pattern):
    img = cv.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image {image_path}")
        exit()

    image_height, image_width, channels = img.shape
    # print(f"Image dimensions: {image_height}x{image_width}, Channels: {channels}")

    # perform object detection
    results = model.predict(source=image_path, conf=0.8, save=True, save_txt=True)

    # print ('YOLO RESULTS: ', results[0].save_dir)

    # get image file name
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # yolo output saved directory
    save_dir = results[0].save_dir

    # path to labels txt file
    labels_file = os.path.join(save_dir, "labels", f"{filename}.txt")

    # if file doesn't exist
    if not os.path.exists(labels_file):
        print(f"The file {labels_file} does not exist.")
        return {"feedback": "Error Occured! Please Try Again."}

    # array to save yolo output
    sorted_yolo_output = []

    sorted_yolo_output = sort_labels(labels_file)

    # class to color mapping
    class_to_color = results[0].names

    # scales normalized bounding box coordinates to pixel values based on image dimensions
    rectangles = [
        (int(x * image_width), int(y * image_height), int(w * image_width), int(h * image_height), class_to_color[cls])
        for cls, x, y, w, h in sorted_yolo_output
    ]

    feedback = ""

    alignment_result = is_horizontally_aligned(rectangles, image_height, 20)
    pattern_result = is_pattern_correct(rectangles, expected_pattern)

    final_score = round(calculate_final_score(alignment_result["alignment_score"], pattern_result["pattern_score"]))

    if alignment_result["is_aligned"]:
        if pattern_result["is_pattern_correct"]:
            feedback = "Rectangles are horizontally aligned. The pattern is built correctly"
        else:
            feedback = "Rectangles are horizontally aligned. The pattern is incorrect"
    else:
        feedback = "Rectangles are not horizontally aligned"
    
    return {"feedback": feedback, "score": final_score}

# Sort objects in label file from left to right
def sort_labels(label_file):
    detections = []

    # read label file
    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split() # split by spaces
            if len(parts) < 5:
                continue  # ignore malformed lines
            
            cls = int(parts[0])  # class ID
            x = float(parts[1])  # x center coordinate
            y = float(parts[2])  # y center coordinate
            w = float(parts[3])  # width of the box
            h = float(parts[4])  # height of the box

            # store as tuple
            detections.append((cls, x, y, w, h))

    # sort detections by x_center (left to right)
    # lambda - to extract the sorting key (x_center) dynamically
    detections.sort(key=lambda item: item[1])
    return detections


# checks all boxes
def is_horizontally_aligned(rectangles, img_height, tolerance_percentage=20):
    # average bounding box height
    avg_box_height = sum(h for _, y, _, h, _ in rectangles) / len(rectangles)
    
    tolerance = avg_box_height * (tolerance_percentage / 100)
    print(f"Tolerance Percentage: {tolerance_percentage}", f"Tolerance: {tolerance}")

    y_positions = [y for _, y, _, _, _ in rectangles]

    misalignment_score = 0
    isAligned = True

    # check if difference between each pair of y-positions is within the tolerance
    for i in range(len(y_positions)):
        for j in range(i + 1, len(y_positions)):
            difference = abs(y_positions[i] - y_positions[j])
            print("i: ", y_positions[i], " j: ", y_positions[j], " difference: ", difference)
            if difference > tolerance:
                print(f"Boxes at indices {i} and {j} are not aligned.")
                misalignment_score += difference
                isAligned = False

    max_possible_misalignment = len(y_positions) * tolerance
    alignment_score = max(0, (1 - misalignment_score / max_possible_misalignment)) * 100  # score between 0 and 100

    # if all boxes are aligned
    if isAligned:
        print("All boxes are aligned.")
    
    return {"is_aligned": isAligned, "alignment_score": alignment_score}

# check pattern
def is_pattern_correct(rectangles, expected_colors):
    detected_colors = [color for _, _, _, _, color in rectangles]
    print('expected color pattern: ', expected_colors)
    print('detected color pattern: ', detected_colors)

    is_pattern_correct = detected_colors == expected_colors

    correct_colors_in_position = 0
    correct_colors_out_of_position = 0
    incorrect_colors = 0

    # check how many colors are correct in position
    for i in range(len(expected_colors)):
        if i < len(detected_colors):
            if detected_colors[i] == expected_colors[i]:
                correct_colors_in_position += 1
            if detected_colors[i] in expected_colors:
                correct_colors_out_of_position += 1
            else:
                incorrect_colors += 1

    total_colors = len(expected_colors)

    correct_position_score = (correct_colors_in_position / total_colors) * 70  
    out_of_position_score = (correct_colors_out_of_position / total_colors) * 30  
    incorrect_penalty = (incorrect_colors / total_colors) * 30

    pattern_score = max(0, correct_position_score + out_of_position_score - incorrect_penalty)

    print("pattern_score: ", pattern_score, " correct_position_score: ", correct_position_score, " out_of_position_score: ",out_of_position_score, " incorrect_penalty: ", incorrect_penalty)

    return {"is_pattern_correct": is_pattern_correct, "pattern_score": pattern_score}

# calculate final score
def calculate_final_score(alignment_score, pattern_score):
    total_score = alignment_score * 0.5 + pattern_score * 0.5
    print("alignment_score: ", alignment_score, " pattern_score: ", pattern_score, " total_score: ", total_score)
    return total_score

# generate random expected color pattern
def generate_pattern(level: int):
    if level not in [2, 3, 4]:
        return {"error", "Invalid Level Selected!"}

    COLORS = ["red-block", "green-block", "blue-block", "yellow-block"]

    pattern = random.sample(COLORS, level)
    return {"pattern": pattern}

# checks only max and min y positions
def is_horizontally_aligned_min_max(rectangles, img_height, tolerance_percentage=5):
    tolerance = img_height * (tolerance_percentage / 100)
    print(tolerance)
    y_positions = [y for _, y, _, _, _ in rectangles]
    val = max(y_positions) - min(y_positions) <= tolerance
    print(max(y_positions) - min(y_positions))
    return val