import cv2

# Load video file
cap = cv2.VideoCapture('input/car_mov1.mp4')

# Load Haar Cascade classifier for car detection
car_cascade = cv2.CascadeClassifier('model/haarcascade_cars.xml')

# Define counting variables
car_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        # End of video
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the image using the Haar Cascade classifier
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in cars:
        # Draw bounding box around car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if car has crossed the counting line
        if y < 400 and y + h > 400:
            car_counter += 1

    # Draw counting line
    cv2.line(frame, (0, 400), (700, 400), (255, 255, 0), 2)

    # Display count on frame
    cv2.putText(frame, f"Count: {car_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()