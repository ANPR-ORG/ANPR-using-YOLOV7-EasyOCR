import easyocr
import torch
import cv2
import numpy as np
import time
import streamlit as st
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from PIL import Image
st.set_page_config(page_title="ANPR", page_icon=":camera:", layout="wide")
st.title('ANPR : Automatic Number Plate Recognition')


def text(img):
    # img = cv2.resize(img, (500, 200))
    # img = cv2.resize(img, (495, 195))
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(img, allowlist='0123456789-.ABCDEFGHIJKLMNOPQRTUVWXYZ')
    st.sidebar.image(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], use_column_width=True)
    str=""
    for ele in ocr_result:
        str += ele[1]+' '
    # return ocr_result[0][1]
    return str


out = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('C:/Users/tejus/PycharmProjects/yolov7_custom/yolov7/best.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()
st.sidebar.title('Settings')
confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, step=0.05)
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], key='image')
if uploaded_file is not None:
    img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    st.sidebar.image(img, caption="Uploaded Image", use_column_width=True)
    st.sidebar.text('Detected Plate:')
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

    output = non_max_suppression_kpt(output, confidence_threshold, 0.65, nc=model.yaml['nc'], kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    res = ""
    for idx in range(output.shape[0]):
        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        plate_roi = nimg[int(ymin):int(ymax), int(xmin):int(xmax)]
        out = text(plate_roi)
        print(out)
        try:
            cv2.putText(nimg, f'{out}', (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX,1, (204, 0, 204), 3)
            cv2.rectangle(nimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (31, 255, 31), 2, cv2.LINE_AA)
        except:
            print("Image not clear")
    st.markdown('''## Result:''')
    ans = st.image(nimg, use_column_width=True)

