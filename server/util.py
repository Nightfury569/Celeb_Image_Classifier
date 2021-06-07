import cv2
import numpy as np
import json
import base64
import joblib
from wavelet import w2d
import sys

__name_to_number = {}
__number_to_name = {}

__model_log_reg = None      # Model Logistic Regression
__model_svm = None          # Model SVM


results_log_reg = []
results_svm = []



def classify_image(image_base64_data,file_path=None):                   # this function gives result using logistic_regression
    imgs = croppedImageofFace_if_2_eyes(file_path,image_base64_data)

    for img in imgs:
        scaled_original_img = cv2.resize(img, (30, 30))
        img_har = w2d(img, 'db1', 5)                        # Wavelength transformed images
        scaled_img_har = cv2.resize(img_har, (30, 30))

        combined_img = np.vstack((scaled_original_img.reshape(30 * 30 * 3, 1), scaled_img_har.reshape(30 * 30, 1)))
        len_image_array = 30 * 30 * 3 + 30 * 30

        final = combined_img.reshape(1, len_image_array).astype(float)


        results_log_reg .append({
            'class': number_to_name(__model_log_reg.predict(final)[0]),
            'class_probability': np.round(__model_log_reg.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __name_to_number
        })
        return results_log_reg



def classify_image_svm(image_base64_data,file_path=None):               # This function gives result using svm model
    imgs = croppedImageofFace_if_2_eyes(file_path,image_base64_data)

    for img in imgs:
        scaled_original_img = cv2.resize(img, (30, 30))
        img_har = w2d(img, 'db1', 5)                        # Wavelength transformed images
        scaled_img_har = cv2.resize(img_har, (30, 30))

        combined_img = np.vstack((scaled_original_img.reshape(30 * 30 * 3, 1), scaled_img_har.reshape(30 * 30, 1)))
        len_image_array = 30 * 30 * 3 + 30 * 30

        final = combined_img.reshape(1, len_image_array).astype(float)

        results_svm.append({
            'class': number_to_name(__model_svm.predict(final)[0]),
            'class_probability': np.round(__model_svm.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __name_to_number
        })
        return results_svm





def load_artifacts():
    print("loading saved artifact...hold on....")

    global __name_to_number
    global __number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __name_to_number = json.load(f)
        __number_to_name = {v:k for k,v in __name_to_number.items()}  # we are getting name of the celebrities from their assigned numbers
                                                                      # Like if number is 0 we are getting cristiano_ronaldo, if number is 1 we are getting lionel_messi
                                                                      # See class_dictionary.json file to understand
    global __model_log_reg
    if __model_log_reg is None:
        with open('./artifacts/saved_model_joblib_logistic.pkl', 'rb') as f:
            __model_log_reg = joblib.load(f)

    global __model_svm
    if __model_svm is None:
        with open('./artifacts/saved_model_joblib_svm.pkl', 'rb') as f:
            __model_svm = joblib.load(f)
    print("loading saved artifacts finished")



def get_cv2_image_from_base64_string(b64str):
    """
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def croppedImageofFace_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/prant/Documents/Jupyter Notebook/Image Classifier/model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/prant/Documents/Jupyter Notebook/Image Classifier/model/opencv/'
                                        'haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    cropped_faces = []

    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if not len(eyes) < 2:
            cropped_faces.append(roi_color)  # Here we are using cropped_faces as there are 2 images of same name it will return 2 images instead of one image

    return cropped_faces


def get_b64_testImage():
    # with open('b64_ronaldo.txt') as f:
    with open('b64_messi.txt') as f:
        return f.read()

def number_to_name(num):
    return __number_to_name[num]

# def name_probabiltiy(image_base64_data,file_path=None):
#     r = classify_image(image_base64_data, file_path)
#     print("name: ",r.get('class') + "\n")
#     print("class-probability: ")
#     for i in range(5):
#         print("\n\tprobability of being " + __number_to_name.get(i) + " : " + str(r.get('class-probability')[0][i]) )
#



if __name__ == '__main__':
    load_artifacts()

    # print(classify_image(None, './test_images/0.jpg'))
    # print(classify_image(None, './test_images/1.jpg'))
    # print(classify_image(None, './test_images/2.jpg'))
    # print(classify_image(None, './test_images/3.jpg'))
    # print(classify_image(None, './test_images/4.jpg'))
    # print(classify_image(None, './test_images/5.jpg'))
    # print(classify_image(None, './test_images/6.jpg'))
    # print(classify_image(None, './test_images/7.jpg'))
    # print(classify_image(None, './test_images/8.jpg'))
    # print(classify_image(None,'./test_images/collage.jpg'))

    # sys.exit()
    # print(classify_image(get_b64_testImage(), None))
    # name_probabiltiy(get_b64_testImage(), None)

    # print(__number_to_name)
    # print(__number_to_name.keys())
    # print(__number_to_name.get(0))