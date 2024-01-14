import cv2
from flask import Flask
from flask_restful import Resource, Api

            # initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)


class PeopleCounter(Resource):
    def get(self):
        # load image
        image = cv2.imread('Zakupy-przedswiateczne-w-PRL.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounter, '/')

if __name__ == '__main__':
    app.run(debug=True)

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
