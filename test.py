import streamlit as st
from joblib import load
import numpy as np

# Sử dụng caching để tải và lưu trữ vectorizer và model
def load_model():
    vectorizer = load('vectorizer.job')
    model = load('model.job')
    return vectorizer, model

# Định nghĩa lại hàm makeTokens ngay tại đây
def makeTokens(f):
    # Đoạn code xử lý và tạo tokens từ URL
    tkns_BySlash = str(f.encode('utf-8')).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')  # make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))  # remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

vectorizer, model = load_model()

# Tạo giao diện người dùng trên web
st.title('Ứng dụng phân loại tên miền')

# Tạo một text input để người dùng nhập URL
domain_to_check = st.text_input('Nhập tên miền cần kiểm tra:', '')

# Khi người dùng nhập URL và nhấn nút 'Phân loại'
if st.button('Phân loại'):
    # Sử dụng hàm makeTokens để xử lý URL nhập vào
    tokens = makeTokens(domain_to_check)

    # Biến đổi URL thành ma trận đặc trưng sử dụng vectorizer
    X_predict = vectorizer.transform([domain_to_check])  # Đảm bảo rằng input là một list

    # Dự đoán xác suất cho mỗi lớp
    probabilities = model.predict_proba(X_predict)

    # Dự đoán kết quả và lấy lớp có xác suất cao nhất
    prediction = model.predict(X_predict)
    max_class_index = probabilities.argmax()
    max_class_label = model.classes_[max_class_index]
    max_probability = probabilities[0, max_class_index]

    # Hiển thị kết quả dự đoán và xác suất lớp cao nhất
    st.write('Kết quả dự đoán:', prediction[0])
    st.write('Xác suất dự đoán:', f"{round(max_probability * 100, 2)}%")

