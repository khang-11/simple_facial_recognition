import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from main import image_to_embedding
from sklearn.preprocessing import Normalizer
import cv2
import time
import sys

def predict(image):

    clf = pickle.load(open("model.sav", 'rb'))

    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

    embedded = image_to_embedding(mtcnn, resnet, image)

    in_encoder = Normalizer(norm='l2')
    embedded = in_encoder.transform(embedded)

    probabilities = clf.predict(embedded)
    print(probabilities)
    '''
    top_n = sorted(range(len(probabilities[0])), key=lambda i: probabilities[0][i])[-20:]

    for i in top_n:
        print(probabilities[0][i], clf.classes_[i])
    '''

def webcam_predict(photo_time = 5):
    cam = cv2.VideoCapture(0)
    for i in range(photo_time, 0, -1):
        print("Taking photo in " + str(i) + " seconds")
        time.sleep(1)

    frame = cam.read()[1]
    cv2.imwrite('face.png', frame)
    del(cam)

    predict('face.png')

# sys.stdout = open("results.txt", "w")
predict('face.jpeg')
