import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Предсказание стоимости квартиры в Москве",
)

@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

model = load_model()

n_rooms = st.sidebar.slider("Выберите количество комнат:", 1, 20, 1)
floor = st.sidebar.number_input("Введите номер этажа:", 1, 100, 1)
square = st.sidebar.number_input("Введите площадь объекта:", 15, 5000, 15)

if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = "Нажмите *Предсказать*, чтобы узнать цену!"

def predict_on():

    input_df = pd.DataFrame({
        "total_square": square,
        "rooms": n_rooms,
        "floor": floor
    }, index =[0])

    price_predicted = int(model.predict(input_df)[0])
    st.session_state.predicted_price = f"Предсказанная стоимость: {int(price_predicted):,} рублей"

left, right = st.columns([0.5, 0.5], vertical_alignment="center")

left.button("Предсказать", on_click = predict_on)
right.markdown(st.session_state.predicted_price)