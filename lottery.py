import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    return df
def prepare_data(numbers):
    X = numbers.iloc[:, :-15].values
    y = numbers.iloc[:, -15:].values
    return X, y
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def generate_numbers(prediction):
    unique_numbers = list(range(1, 26))
    predicted_numbers = []
    for num in prediction:
        chosen_number = max(1, min(25, round(num)))
        if chosen_number in unique_numbers:
            predicted_numbers.append(chosen_number)
            unique_numbers.remove(chosen_number)
    while len(predicted_numbers) < 15:
        chosen_number = random.choice(unique_numbers)
        predicted_numbers.append(chosen_number)
        unique_numbers.remove(chosen_number)
    predicted_numbers.sort()
    return predicted_numbers
def main():
    file_path = 'Teste.v2.xlsx'
    sheet_name = 'Planilha1'
    data = load_data(file_path, sheet_name)
    numbers = data
    X, y = prepare_data(numbers)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    next_draw = model.predict(X_test[-1].reshape(1, -1))
    predicted_numbers = generate_numbers(next_draw.flatten())
    print("Previsão para o próximo sorteio:")
    print(predicted_numbers)
if __name__ == "__main__":
    main()
