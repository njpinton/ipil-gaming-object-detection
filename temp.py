
"""
TODO

-> argparse
-> how to return image with bounding boxes
-> option to save image?
-> get help? haha nope
-> i've seen other codes with function for creating bounding box
    -> do you want to do that as well?
-> oh noes you still haven't done any documentation at all,
    are you gonna be alright? hahaha
-> redesign main function

Notes:

w and h of x,y,w,h of mtcnn and haar have different definition,
    -> do you want to standardize them as well?

"""


import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from datetime import datetime
import argparse
import numpy as np
import cv2
import os, time, glob
import pickle
import multiprocessing as mp

from align import detect_face
import facenet
from classifier import classify_rf_predict as lr_predict


threshold_face = 0.5

FREQ_DIV = 20
RESIZE_FACTOR = 4
CHECKPOINT_DIV = 60
NUM_TRAINING = 12

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("models/20180408-102900.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

# Get labels
labels = pd.read_csv('labels.txt', header=None, names=['label', 'name'])

# Emotion Detection
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# Gender detection
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_classifier = load_model(gender_model_path, compile=False)
gender_target_size = gender_classifier.input_shape[1:3]
gender_labels = ["woman", "man"]

# Age prediction
age_model = 'trained_models/age_models/age_net.caffemodel'
age_proto = 'trained_models/age_models/deploy_age.prototxt'
age_net = cv2.dnn.readNet(age_model, age_proto)
age_list = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)',
            '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
MODEL_MEAN_VALUES = [104, 117, 123]


class RecogFaces:
    def __init__(self):
        cascade_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.face_dir = 'face_images'
        self.face_images_dir = os.path.join(self.face_dir,'images')
        self.face_logs_dir = os.path.join(self.face_dir,'logs')
        self.image_dir = 'screenshot_images'
        self.image_images_dir = os.path.join(self.image_dir,'images')
        self.image_logs_dir = os.path.join(self.image_dir,'logs')
        self.embedding_dir = 'embeddings'
        # creates new path for new subject
        create_folder(self.image_images_dir)
        create_folder(self.face_images_dir)
        create_folder(self.embedding_dir)
        self.image_captures = 0
        self.count_captures = 0
        self.count_timer = 0
        self.labels = pd.read_csv('labels.txt', header=None, names=['label', 'name'])
        self.lr = pickle.load(open('model_lr_lbfgs.sav', 'rb'))
        # print(self.labels)
        self.log_path = os.path.join(self.face_logs_dir,'logs.csv')
        if not os.path.isfile(self.log_path):
            pd.DataFrame(columns=['date','time']).to_csv(self.log_path,index=False)
        self.logs = pd.read_csv(self.log_path)

    def capture_video(self, face_detect='mtcnn', save_images=False, recognize=True,
                      emotion=True, gender=True, age=True, is_train=False, video=0):

        video_capture = cv2.VideoCapture(video)
        print(face_detect)
        while True:
            self.count_timer += 1
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            inImg = np.array(frame)

            self.process_image(inImg, face_detect, save_images, recognize,
                               emotion, gender, age)

            if is_train:
                save_images = True
                img_no = sorted([int(fn[:fn.find('.')]) for fn in \
                        os.listdir(self.face_images_dir) if fn[0]!='.' ]+[0])[-1] + 1
                if img_no > NUM_TRAINING:
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return

            cv2.imshow('Video',inImg)
            # q to quit or to finish training and go back to main menu
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logs.to_csv(self.log_path, index=False)
                video_capture.release()
                cv2.destroyAllWindows()
                return

    def capture_image(self, img, fname, face_detect='mtcnn', save_images=True, recognize=True,
                      emotion=True, gender=True, age=True):

        inImg = np.array(img)
        self.process_image(inImg, face_detect, save_images, recognize,
                           emotion, gender, age)

        # cv2.imshow('Faces Found',inImg)
        # img_no = sorted([int(fn[:fn.find('.')]) for fn in \
        #                  os.listdir(self.image_images_dir) if fn[0] != '.'] + [0])[-1] + 1
        cv2.imwrite('%s/%s.png' % (self.image_images_dir, fname), inImg)
        self.logs.to_csv(self.log_path, index=False)
        cv2.waitKey(0)
        return

    def process_image(self, img, face_detect, save_images, recognize,
                      emotion, gender, age):

        # detect faces in frame
        if face_detect == 'mtcnn':
            faces = self.detect_face_mtcnn(img)
        else:
            faces = self.detect_face_haar(img)

        # perform recognition, emotion classification, gender classification, and age classification
        names = []
        emotions = []
        genders = []
        ages = []
        name_probs = []
        for face in faces:
            if recognize:
                name, name_prob = self.recognize_face(face['face'], face['rect'], img)
            else: name, name_prob = ("",0)
            names.append(name)
            name_probs.append(name_prob)

            if emotion:
                emotion, emotion_prob = self.detect_emotion(face['face'], face['rect'], img)
            else: emotion, emotion_prob = ("", 0)
            emotions.append(emotion)

            if gender:
                gender, gender_prob = self.detect_gender(face['face'], face['rect'], img,
                                                         gender_target_size, gender_labels)
            else: gender, gender_prob = ("", 0)
            genders.append(gender)

            if age:
                age, age_prob = self.detect_age(face['face'], face['rect'], img, age_list)
            else: age, age_prob = ("", 0)
            ages.append(age)

            # save faces in frames
            if save_images:
                if self.count_timer % FREQ_DIV == 0:
                    img_no = self.write_img(face['face'])
                    self.write_log(img_no, name, name_prob, emotion, emotion_prob,
                                   gender, gender_prob, age, age_prob, face['rect'])


        # put bounding boxes around faces
        for face,name,name_prob,emotion,gender,age in zip(faces,names,name_probs,emotions,genders,ages):
            x = face['rect'][0]
            y = face['rect'][1]
            w = face['rect'][2]
            h = face['rect'][3]
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)
            if emotion:
                cv2.putText(img, 'Emotion : {}'.format(emotion), (x - 10, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if gender:
                cv2.putText(img, gender, (x - 10, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if age:
                cv2.putText(img, age, (x + 30, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if recognize:
                self.write_name(img,face['rect'],name,name_prob,threshold_face)

        # cv2.imwrite(output_image+'.jpg', img)

    def detect_face_haar(self,img):
        faces = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bounding_boxes = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                            minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                (x, y, w, h) = face
                cropped = img[y:y+h, x:x+w]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                faces.append({'face': cropped, 'rect': [x,y,w,h]})
        return faces

    def detect_face_mtcnn(self,img):
        faces = []
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if not len(bounding_boxes) == 0:
            for face in bounding_boxes:
                if face[4] > 0.75:
                    det = np.squeeze(face[0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    x = bb[0]
                    y = bb[1]
                    w = bb[2]
                    h = bb[3]
                    faces.append({'face': cropped, 'rect': [x,y,w,h]})
        return faces

    def recognize_face(self, face, bb, img):
        resized = cv2.resize(face, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
        embed = self.get_embedding(resized)
        name, score = self.get_name(embed)
        self.write_name(img,bb,name,score,threshold_face)
        return name, score

    def resize_face(self,face):
        """should be incorporated to self.get_embeddings"""
        return cv2.resize(face, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)

    def write_name(self,img,bb,name,score,threshold):
        (x, y, w, h) = bb
        if score > threshold:
            # text = 'Face Detected'
            text = 'name: {}'.format(name)
        else: text = 'Face Detected'
        cv2.putText(img, text, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    def write_img(self, cropped):
        img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.face_images_dir) if fn[0]!='.' ]+[0])[-1] + 1
        cv2.imwrite('%s/%s.png' % (self.face_images_dir, img_no), cropped)
        return img_no

    def detect_emotion(self, face, rect, img):
        roi = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (64,64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        return label, emotion_probability

    def detect_gender(self, face, rect, img, gender_target_size, gender_labels):
        rgb_face = cv2.resize(face, (gender_target_size))
        rgb_face = rgb_face.astype("float") / 255.0
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_probability = np.max(gender_prediction[0])
        return gender_text, gender_probability

    def detect_age(self, face, rect, img, age_list):
        MODEL_MEAN_VALUES = [104, 117, 123]
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        return age, age_preds[0].max()

    def resize_face(self, face):
        return cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)

    def get_embedding(self, resized):
        reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
        embedding = sess.run(embeddings, feed_dict=feed_dict)
        return embedding

    def get_name(self,embed):
        y_pred, score = lr_predict(embed, self.lr)
        name = self.labels['name'][self.labels['label'] == y_pred[0]].values[0]
        print(name,score[0])
        return name, score[0]

    def write_log(self, img_no, name='', name_prob=0, emotion='', emotion_prob=0,
                                    gender='', gender_prob=0, age='', age_prob=0, coords=''):
        """append date, time, image filename, prediction, score"""
        now = datetime.now()
        self.logs = self.logs.append({'date': now.strftime("%Y-%m-%d"),
                                      'time': now.strftime("%H:%M:%S"),
                                      'photo_id':"%s.png"%img_no,
                                      'name': name,
                                      'name_prob': "{:.2f}".format(name_prob),
                                      'emotion': emotion,
                                      'emotion_prob': "{:.2f}".format(emotion_prob),
                                      'gender': gender,
                                      'gender_prob': "{:.2f}".format(gender_prob),
                                      'age': age,
                                      'age_prob': "{:.2f}".format(age_prob),
                                      'coords': coords,
                                      },
                                      ignore_index=True)

    def create_output_directory(self, input_dir, output_dir):
       if not os.path.exists(output_dir): os.makedirs(output_dir)
       for image_dir in os.listdir(input_dir):
           image_name = os.path.basename(os.path.basename(image_dir))[:-4]
           image_output_dir = os.path.join(output_dir, image_name)
           if not os.path.exists(image_output_dir):
               os.makedirs(image_output_dir)
       #logger.info('Read {} images from dataset'.format(len(os.listdir(input_dir))))

    # @staticmethod
    # def log_result(self, result):
    #     print('called')
    #     results.append(result)

    def detect_faces(self, input_dir, output_dir):
       result_list = []
       start_time = time.time()
       pool = mp.Pool(processes = (mp.cpu_count()))
       self.create_output_directory(input_dir, output_dir)
       image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
       for index, image_path in enumerate(image_paths):
           output_path = os.path.join(output_dir, os.path.basename(image_path)[:-4])
           img = cv2.imread(image_path)
           img = np.array(img)
           pool.apply_async(self.process_image, (img, 'mtcnn', 1, 1, 1, 1, 1, output_path), callback = self.log_result)
           #result_list.append(result.get())
       pool.close()
       pool.join()
       print('Pre-processed {} images from dataset'.format(len(image_paths)))
       print('Pre-processing: Completed in {} seconds'.format(time.time() - start_time))
       #logger.info('Pre-processed {} images from dataset'.format(len(image_paths)))
       #logger.info('Pre-processing: Completed in {} seconds'.format(time.time() - start_time))


def get_distance(embed1, embed2):
    return np.sqrt(np.sum(np.square(np.subtract(embed1, embed2))))


def get_cosine_similarity(embed1, embed2):
    a = np.matmul(np.transpose(embed1), embed2)
    b = np.sum(np.multiply(embed1, embed2))
    c = np.sum(np.multiply(embed1, embed2))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


import cv2
def play_video(path):
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        cv2.imshow('vid', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return
        

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', default=0, help='uhmm, path of video file, put 0 for camera i think')
    args = parser.parse_args()

    video = args.video_path

    recognizer = RecogFaces()
    recognizer.capture_video(video=video)

    #recognizer.process_image('./vids/test/thumb1.jpg', 'mtcnn', 1, 1, 1, 1, 1, './vids/results/sample.jpg')
    # recognizer.detect_faces(input_dir='./vids/test/', output_dir='./vids/results/')

    #
    # path = 'vids/test'
    # frame_count = 0
    # process_count = 0
    # skip = 15
    # # pool = mp.Pool(processes=(mp.cpu_count()))
    # start = time.time()
    # for img_path in os.listdir(path):
    #     if frame_count % skip == 0:
    #         img = cv2.imread(os.path.join(path,img_path))
    #         recognizer.capture_image(img, img_path[:-4])
    #         process_count += 1
    #     frame_count += 1
    #     print(f"frame {frame_count}, {process_count/(time.time()-start):.2f} fps, {frame_count/(time.time()-start):.2f}")
    # recognizer.logs.to_csv('unilab_logs.csv', index=False)
    # print(f"Processed {process_count} of {frame_count} frames of in {(time.time()-start):.2f} seconds ({skip})")
