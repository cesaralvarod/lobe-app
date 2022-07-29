import argparse
import os
from classes.TFModel import TFModel
from utils.telegram import *
from PIL import Image
import cv2 as cv
import sys
import math

filename = "prediction.png"
to_predict = "Con plaga"
threshold = 0.9
threshold_video = 0.95


def open_image(image="", predictionText="", prediction={}):
    img = cv.imread(cv.samples.findFile(image))

    if img is None:
        sys.exit("Could not read the image")

    w, h, x = img.shape

    cv.putText(img, predictionText, (0, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv.LINE_4)

    cv.imshow("Display window", img)
    cv.imwrite(filename, img)

    print(prediction)

    if prediction['label'] == to_predict and prediction['confidence'] >= threshold:
        caption = f"Se encontró mango {prediction['label']} con un acierto de {prediction['confidence']*100}%."
        send_image_telegram(filename, caption)

    k = cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select an option.")
    parser.add_argument("-i", dest="image", help="Path to your image file.")
    args = parser.parse_args()
    model_dir = os.path.join(os.getcwd(), "")
    model = TFModel(model_dir=model_dir)

    if not args.image:
        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera.")
            exit()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting...")
                break

            cv.imwrite(filename, frame)

            image = Image.open(filename)
            outputs = model.predict(image)

            predictions = outputs["predictions"]
            predictionText = ""

            if predictions:
                prediction = {}
                for pred in predictions:
                    if pred["confidence"] >= threshold_video:
                        predictionText = f"{pred['label']}: {math.trunc(pred['confidence']*100)}%"

                        if pred['label'] == to_predict:
                            send_image_telegram(filename, predictionText)

            cv.putText(frame, predictionText, (0, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_4)

            cv.imshow("Frame", frame)

            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    else:
        if os.path.isfile(args.image):
            image = Image.open(args.image)
            outputs = model.predict(image)

            predictions = outputs["predictions"]

            if predictions:
                predictionText = ""
                prediction = {}
                for pred in predictions:
                    if(pred["confidence"] >= threshold):
                        prediction = pred
                        predictionText = f"{pred['label']}: {math.trunc(pred['confidence']*100)}%"
                open_image(args.image, predictionText, prediction)
        else:
            print("Could not find image.")
