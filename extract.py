import cv2
import numpy as np
import os
import glob
import easyocr
import matplotlib.pyplot as plt

# Function to load image
def load_local_image(path):
    return cv2.imread(path)

# Function to detect black circular patches
def detect_black_circular_patches(img, min_radius=15, max_radius=40, fill_ratio=0.6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to black and white (binary)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert image (so black becomes white)
    inverted = cv2.bitwise_not(binary)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=2*min_radius,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    positions = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            mask = np.zeros_like(binary)
            cv2.circle(mask, (x, y), r, 255, -1)
            circle_pixels = binary[mask == 255]

            black_pixels = np.sum(circle_pixels == 0)
            total_pixels = len(circle_pixels)
            if total_pixels == 0:
                continue

            if black_pixels / total_pixels >= fill_ratio:
                positions.append((x, y))

    return positions, binary  # Also return binary for display

# Function to draw circles on image
def draw_detected_circles(img, positions, radius=10):
    for (x, y) in positions:
        cv2.circle(img, (x, y), radius, (128), 5)  # Use gray circle
    return img

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# Function to get text positions using EasyOCR
def get_text_positions(image,last=False):
    show_image_with_matplotlib(image,"")
    reader = easyocr.Reader(['en'])  # Initialize OCR reader for English
    result = reader.readtext(image,min_size=10,text_threshold=0.4,paragraph=False)
    print(result)
    if not result:
        return None

    # Get the first and last text position (from top to bottom)
    first_text_position = result[0][0]
    last_text_position = result[-1][0]
    
    return first_text_position if not last else last_text_position

# Function to show images with Matplotlib (for environments without OpenCV GUI support)
def show_image_with_matplotlib(image, window_name="Image"):
    # Convert the image from BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Show the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title(window_name)
    plt.show()

# Process each image in the directory
def process_directory(directory_path):
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    for image_path in sorted(image_files):
        print(f"\nProcessing: {os.path.basename(image_path)}")
        image = load_local_image(image_path)
        
        # Crop the top 10% and bottom 10% of the image for text detection
        height, width, _ = image.shape
        top_crop = image[:int(height * 0.2), :]  # Top 10%
        bottom_crop = image[int(height * 0.8):, :]  # Bottom 10%
        top_crop = preprocess_for_ocr(top_crop)
        bottom_crop = preprocess_for_ocr(bottom_crop)
               

        show_image_with_matplotlib(bottom_crop,"Bottom Crop")


        # Get text positions from cropped images
        first_text_position, last_text_position = None, None
        
        first_text_position = get_text_positions(top_crop)
        
        last_text_position = get_text_positions(bottom_crop,True)
        
        print("First text position (top):", first_text_position)
        print("Last text position (bottom):", last_text_position)

      
        # Draw and show results (black circular patches)
        positions, binary = detect_black_circular_patches(image)
        print("Detected circular black spots at:", positions)

        image_with_circles = draw_detected_circles(binary.copy(), positions)

        if first_text_position:
            top_y_coords = [point[1] for point in first_text_position]
            top_y = int(np.mean(top_y_coords))
            cv2.line(image_with_circles, (0, top_y), (width, top_y), (0, 255, 0), thickness=6)  # Green line for first text

        if last_text_position:
            bottom_y_coords = [point[1] + int(height * 0.9) for point in last_text_position]  # Offset for cropped bottom
            bottom_y = int(np.mean(bottom_y_coords))
            cv2.line(image_with_circles, (0, bottom_y), (width, bottom_y), (255, 0, 0), thickness=6)  # Red line for last text


        resized_image_with_circles = cv2.resize(image_with_circles, (1000, 1000))
        show_image_with_matplotlib(resized_image_with_circles, "Detected Black Circles (B&W)")

    

# === USAGE ===
directory_path = "../process-data/sheets_images/er_posttest_lab_renamed"  # Replace with your directory
process_directory(directory_path)
