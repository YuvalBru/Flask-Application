
# קובץ פייתון - api.py:

# נשתמש באפליקציית Flask לקבלת נתוני הרכב, ביצוע העיבוד והחיזוי והחזרת התוצאה:
#  הסבר קוד –
#  בקובץ זה אנו יוצרים את אפליקציית Flask .
# •  ייבוא ספריות: נייבא את Flask, pandas, joblib, והפונקציה prepare_data.
# •  הגדרת האפליקציה: ניצור את אפליקציית Flask ונטען את המודל המאומן.
# נhome  (/): מציגה את דף ה-HTML.
# •  (/predict): מקבלת את הנתונים מהטופס, מכינה אותם בעזרת prepare_data, ומחזירה את החיזוי מהמודל.
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import joblib
from car_data_prep import prepare_data

app = Flask(__name__)
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Year': int(request.form['year']),
        'Gear': request.form['gear'],
        'Engine_type': request.form['engine_type'],
        'Prev_ownership': request.form['prev_ownership'],
        'Curr_ownership': request.form['curr_ownership'],
        'Area': request.form['area'],
        'Color': request.form['color'],
        'manufactor': request.form['manufactor'],
        'model': request.form['model'],
        'Hand': int(request.form['hand']),
        'Km': float(request.form['km']),
        'capacity_Engine': int(request.form['capacity_engine']),
        'Cre_date': request.form['cre_date'],
        'Repub_date': None,
        'Description': None,
        'Test': None,
        'Pic_num': None,
        'Supply_score': None,
        'City': None,
        'Price': np.nan
    }
    df = pd.DataFrame(data, index = [0])
    df['Engine_type'] = df['Engine_type'].astype('category')
    df['Engine_type'] = df['Engine_type'].cat.add_categories(['גז', 'היברידי'])
    print(df['Engine_type'])
    df_prepared = prepare_data(df)
    X = df_prepared.drop(columns=['Price', 'Cre_date'])
    prediction = model.predict(X)
    return f'מחיר הרכב החזוי: {prediction[0]:.2f}'

if __name__ == '__main__':
    app.run(debug=True)