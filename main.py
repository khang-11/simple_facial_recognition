from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os
import sys


def image_to_embedding(mtcnn, resnet, img_path):
    # open image
    img = Image.open(img_path)

    # crop image
    img_cropped = mtcnn(img)

    # use facenet to embed image to 8631 dimension vectors
    img_probs = resnet(img_cropped.unsqueeze(0))

    # detach gradient and return as numpy array
    no_gradient_embedding = img_probs.detach()
    embedding = no_gradient_embedding.numpy()

    return embedding

def embed_folder(mtcnn, resnet, folder, name, npz = None):
    face_embeddings = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".JPG"):
            try:
                face_embeddings.append(image_to_embedding(mtcnn, resnet, folder + filename))
                print(filename + " loaded")
            except:
                print(filename + " BAD BITCH")

    if npz == None:
        np.savez('faces.npz', face_embeddings, [name for _ in range(len(face_embeddings))])

    else:
        x = np.load(npz)
        combined_face_embeddings = np.concatenate((x['arr_0'], face_embeddings))
        combined_face_names = np.concatenate((x['arr_1'], [name for _ in range(len(face_embeddings))]))
        np.savez(npz, combined_face_embeddings, combined_face_names)

def embed_all(directory, new = True):
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
     
    training = os.listdir(directory)

    try:
        training.remove(".DS_Store")
    except:
        ""

    if new:
        embed_folder(mtcnn, resnet, directory + "/" + training[0] + "/", training[0]) 
        for i in range(1, len(training)):
            embed_folder(mtcnn, resnet, directory + "/" + training[i] + "/", training[i], npz="faces.npz")

    else:
        for i in range(0, len(training)):
            embed_folder(mtcnn, resnet, directory + "/" + training[i] + "/", training[i], npz="faces.npz")

if __name__ == "__main__":
    '''
    people = ['obama', 'khang', 'kanye', 'phuong', 'michael', 'randy', 'serena']

    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

    embed_folder(mtcnn, resnet, "faces/" + people[0] + "/", people[0])
    for i in range(1, len(people)):
        embed_folder(mtcnn, resnet, "faces/" + people[i] + "/", people[i], npz = 'faces.npz')
    '''
    embed_all("new")
