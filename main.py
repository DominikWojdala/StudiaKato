import cv2
from flask import Flask, request
from flask_restful import Resource, Api
import os
import requests

#image_url = "https://d-art.ppstatic.pl/kadry/k/r/1/f8/d5/84dcc8f0709d66e6f7c22593b61d_o_original.jpg"
#img_data = requests.get(image_url).content

# Specify the desired directory and filename
#save_path = os.path.join(r"C:\Users\student\StudiaKato", 'zony.jpg')



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCounterStatic(Resource):
    def get(self):
        # load image
        image = cv2.imread('zony.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


class PeopleCounterDynamicUrl(Resource):
    def get(self):

        # TODO:
        # 1. Pobrać zdjęcie z otrzymanego adresu
        # 2. Pobrane zdjęcie można zapisać na dysku lub przetwarzać je w pamięci podręcznej
        # 3. Załadowane zjęcie do zmiennej image przekazać do algorytmu hog.detectMultiScale i zwrócić z endpointu liczbę wykrytych osób.

        url = request.args.get('url')
        print('url', url)

        img_data = requests.get(url).content
        #cv2.imshow("People detector", img_data)
        save_path = os.path.join(r"C:\Users\student\StudiaKato", 'zony.jpg')
        with open(save_path, 'wb') as handler:
            handler.write(img_data)
        image = cv2.imread('zony.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}



api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)

'''import cv2
from flask import Flask, request
from flask_restful import Resource, Api

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCounter(Resource):
    def get(self):
        url = request.args.get('url:')
        print('url:', url)
        # load image
        image = cv2.imread('')
        image = cv2.resize(image, (700, 400))
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounter, '/')

if __name__ == '__main__':
    app.run(debug=True)
'''
'''import cv2
from flask import Flask
from flask_restful import Resource, Api


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load image
image = cv2.imread('Zakupy-przedswiateczne-w-PRL.jpg')
image = cv2.resize(image, (700, 400))

# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# draw the bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(f'Found {len(rects)} humans')

# show the output images
cv2.imshow("People detector", image)
cv2.waitKey(0)
'''
'''
img = cv2.imread('Zakupy-przedswiateczne-w-PRL.jpg')
im = cv2.resize(img,(500,300))

print(type(img))
print(img.shape)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
