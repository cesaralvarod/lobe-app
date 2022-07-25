import argparse
import os
from classes.TFModel import TFModel
from PIL import Image
import cv2 as cv
import sys


def open_image(image="", prediction=""):
    img = cv.imread(cv.samples.findFile(image))

    if img is None:
        sys.exit("Could not read the image")
    w, h, x = img.shape
    cv.putText(img, prediction, (0, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv.LINE_4)
    cv.imshow("Display window", img)
    k = cv.waitKey(0)


if __name__ == "__main__":
    threshold = 0.9

    parser = argparse.ArgumentParser(
        description="Select an option.")
    parser.add_argument("-i", dest="image", help="Path to your image file.")
    args = parser.parse_args()
    model_dir = os.path.join(os.getcwd(), "")

    if not args.image:
        print("Con camara")
    else:
        if os.path.isfile(args.image):
            image = Image.open(args.image)
            model = TFModel(model_dir=model_dir)
            outputs = model.predict(image)

            predictions = outputs["predictions"]

            if predictions:
                prediction = ""
                for pred in outputs["predictions"]:
                    if(pred["confidence"] > threshold):
                        prediction = f"{pred['label']} : {round(pred['confidence'], 2)*100}%"
                open_image(args.image, prediction)
        else:
            print("Could not find image.")
