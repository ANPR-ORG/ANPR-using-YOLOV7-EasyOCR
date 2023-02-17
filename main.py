import easyocr
import torch
import cv2
import numpy as np
import time

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def text(img):
    img = cv2.resize(img, (500, 200))
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ012346789.-')
    # cv2.imshow('1', cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    # cv2.waitKey(0)
    str=""
    for ele in ocr_result:
        str += ele[1]+' '
    # return ocr_result[0][1]
    return str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7/best.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

img_path = 'yolov7/custom_dataset/test/images/00d9db3d2c186504_jpg.rf.a7ce9b40192331cb9c7c7f49cd07f720.jpg'
# img_path = 'yolov7/custom_dataset/test/images/0123f836ce94678a_jpg.rf.ed2d6fcfd88d8c65fd316a5c38c39223.jpg'
# img_path = 'yolov7/custom_dataset/test/images/11-Guerrero-2BPlaca-2Bcapacidades-2Bdiferentes-2B99-GAA-2BFranja-2Babajo_jpg.rf.1653b0c3043fde503b070b34685a42ac.jpg'
img = cv2.imread(img_path)

# Get the frame width and height.
h, w, c = img.shape
frame_width = w
frame_height = h


orig_image = img
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = letterbox(image, (frame_width), stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))
image = image.to(device)
image = image.half()

with torch.no_grad():
    output, _ = model(image)

output = non_max_suppression_kpt(output, 0.2, 0.65, nc=model.yaml['nc'], kpt_label=True)
output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)


for idx in range(output.shape[0]):
    # plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    # Comment/Uncomment the following lines to show bounding boxes around persons.
    xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
    xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

    plate_roi = nimg[int(ymin):int(ymax), int(xmin):int(xmax)]
    try:
        out = text(plate_roi)
        print(out)
        cv2.putText(nimg, f'{out}', (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)

    except:
        print("Image not clear")
    cv2.rectangle(
        nimg,
        (int(xmin), int(ymin)),
        (int(xmax), int(ymax)),
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA
    )

# Convert from BGR to RGB color format.
# cv2.imwrite('result.jpg',nimg)
cv2.imshow('result', nimg)
cv2.waitKey(0)