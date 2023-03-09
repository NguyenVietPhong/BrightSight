import pickle
import os 
import face_recognition


def load_image_face(img_path):
    name_image = img_path.split('/')[-1].split('.')[0] 

    return name_image, face_recognition.load_image_file(img_path)

def _feature_face(image):
    return face_recognition.face_encodings(image)[0]

def save_feature_face(pickle_path, obj):

    with open(pickle_path, 'ab+') as f:
        pickle.dump(obj, f)
        f.close()



def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        obj = pickle.load(f)

        return obj
    

def main_save(path, pickle_path, type='file'):
    obj=[]
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        name_image, image = load_image_face(img_path)
        feature_face = _feature_face(image)
        obj_face = {'name': name_image, 'feature_face': feature_face}
        obj.append(obj_face)

    print(obj)
    save_feature_face(pickle_path, obj)


def main():
    pickle_path = "/home/phong/system_project/face_recognition/face_recognition_face.pkl"
    path = "/home/phong/system_project/face_recognition/images_face"
    main_save(path, pickle_path)

    # load_pickle(pickle_path)

main()

