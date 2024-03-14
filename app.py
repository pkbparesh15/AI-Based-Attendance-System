from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import time

model_file = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_file = 'deploy.prototxt.txt'

conf_min = 0.2 #Confidence level for face detection

#Image Quality check parameters
blur_threshold = 100  # Adjust as needed
exposure_threshold_low = 50  # Adjust as needed
exposure_threshold_high = 150  # Adjust as needed
min_face_count_warning = 9  # Adjust as needed

#Denoising parameters
h_luminance = 8  # Decreasing the strength of noise reduction for luminance.
h_color = 8  # Decreasing the strength of noise reduction for color components.
template_win_size = 7  # This can stay the same or be reduced if the process is too aggressive.
search_win_size = 21  # This can also stay the same or be reduced for increased speed.


app = Flask(__name__)

def enhance_image(image):
    # Histogram equalization for improving the contrast of the image
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    enhanced_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    # Denoising the image
    enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, h_luminance, h_color, template_win_size, search_win_size)

    
    return enhanced_img

def is_blurry(image):
    # Simple blur detection using Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < blur_threshold

def is_overexposed(image):
    # Simple exposure check based on average pixel intensity
    intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    return intensity > exposure_threshold_high

def is_underexposed(image):
    # Simple exposure check based on average pixel intensity
    intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    return intensity < exposure_threshold_low

@app.route('/attendance')
def index():
    return render_template('index.html')

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        (h, w) = image.shape[:2]

        # Image enhancement
        image = enhance_image(image)

        # Quality checks
        if is_blurry(image):
            return jsonify({'error': 'Image is too blurry. Please reload.'})

        if is_overexposed(image):
            return jsonify({'error': 'Image is overexposed. Please reload.'})

        if is_underexposed(image):
            return jsonify({'error': 'Image is underexposed. Please reload.'})

        # Continue with face detection
        start = time.time()
        network = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (900, 900)), 1.0, (900, 900), (104.0, 117.0, 123.0))
        network.setInput(blob)
        detections = network.forward()

        # Draw rectangles around the faces
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_min:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_min:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # Clamping box coordinates to the image dimensions
                start_x = max(0, min(start_x, w - 1))
                start_y = max(0, min(start_y, h - 1))
                end_x = max(0, min(end_x, w - 1))
                end_y = max(0, min(end_y, h - 1))

                color = image[start_y, start_x]
                if color[1] > color[0] and color[1] > color[2]:
                    count += 1


        # Save the image with rectangles (for demonstration purposes)
        cv2.imwrite('static/result_image.jpg', image)

        # Check for the minimum face count and add a warning if needed
        warning_message = None
        if count < min_face_count_warning:
            warning_message = f"Warning: The Attendance ({count}) is less than the expected minimum attendance ({min_face_count_warning})."

        # Return the number of detected faces along with the warning message
        return jsonify({'result': count, 'warning': warning_message})

if __name__ == '__main__':
    app.run(debug=True)
