import cv2
import numpy as np

# Load the image
image = cv2.imread("img/2.png")  # Replace with a screenshot or canvas capture
# print(type(image))  # Should be <class 'numpy.ndarray'> if loaded correctly

if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    # Detect Shape Based on Number of Sides
    if len(approx) == 3:
        shape_name = "Triangle"
    elif len(approx) == 4:
        aspect_ratio = w / float(h)
        shape_name = "Square" if 0.95 < aspect_ratio < 1.05 else "Rectangle"
    elif len(approx) > 4:
        shape_name = "Circle" if cv2.isContourConvex(approx) else "Polygon"
    else:
        shape_name = "Unknown"

    # Draw and Label
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show Image
cv2.imshow("Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
