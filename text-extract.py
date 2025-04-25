import cv2
import streamlit as st
import pytesseract as pyy
import numpy as np

st.set_page_config(page_title="Simple Automated Personal Loan Processing",layout="centered")
st.title("Simple Automated Personal Loan Processing")
upload_file=st.file_uploader("upload the loan doc image",type=["jpg","jpeg","png"])
if upload_file:
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img,caption="Uploaded document",use_container_width=True)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    _,thresh=cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
    cv2.imwrite("temp/binary.jpg",thresh)

    filtered=cv2.medianBlur(thresh,1)
    cv2.imwrite("temp/fileterd-img.jpg",filtered)

    oc= pyy.image_to_string(filtered)

    st.subheader("Extracted Text")
    st.text_area("OCR-Result",oc,height=150)



