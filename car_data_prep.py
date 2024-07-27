# קובץ פייתון- car_data_prep.py:
# נשתמש בפונקציית prepare_data כפי שהגדרנו אותה:
# קצת הסבר –
# הסבר הפונקציה prepare_data:
# 1.	הסרת עמודות לא רלוונטיות: נסיר עמודות שלא משמשות בחיזוי.
# 2.	המרת סוגי נתונים: נמיר עמודת Cre_date לתאריך והמרת עמודת Price לפורמט מספרי.
# 3.	חישוב גיל הרכב: נחשב גיל הרכב על פי שנת הייצור ותאריך ההכנה.
# וכן הלאה, הסברים יותר מלאים מצויים במטלה 2
import random
import pandas as pd


def prepare_data(df):
    df = df.drop([
        'Repub_date', 'Description', 'Test', 'Pic_num', 'Supply_score', 'City'],
        axis=1
    )
    df['Cre_date'] = pd.to_datetime(df['Cre_date'], errors='coerce')
    df['car_age'] = df['Cre_date'].dt.year - df['Year']
    integer_col = ['Km', 'car_age', 'capacity_Engine']
    cat_col = ['Year', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'Color', 'manufactor',
               'model', 'Hand']

    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

    for col in integer_col:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    df['Feature_Comb1'] = (df['Km'] / (df['car_age'] * 1000 + 10000)) - df['Hand'] * 100
    df['Feature_Comb2'] = df['car_age'] / df['Hand']
    df[cat_col] = df[cat_col].astype('category')
    df['Feature_Comb1'] = df['Feature_Comb1'].astype(float)
    df['Feature_Comb2'] = df['Feature_Comb2'].astype(float)
    df['Curr_ownership'] = df['Curr_ownership'].replace('חברה', 'רה')
    df['Curr_ownership'] = df['Curr_ownership'].replace('חברה', 'חב')
    df['Curr_ownership'] = df['Curr_ownership'].replace('לא מוגדר', None)

    df['Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    df['Km'].fillna(0, inplace=True)
    df['Km'] = df['Km'].apply(lambda x: x * 1000 if 1 <= x <= 1000 else x)
    df['Km'].fillna(df['Km'].median(), inplace=True)
    df['Km'] = df['Km'].replace(0, df['Km'].median())

    df['Engine_type'] = df['Engine_type'].replace('היבריד', 'היברידי')
    df.loc[df['Engine_type'].isna() & (df['capacity_Engine'] > 2250), 'Engine_type'] = 'גז'
    df.loc[df['Engine_type'].isna() & (df['capacity_Engine'] <= 2250), 'Engine_type'] = 'היברידי'
    df['Curr_ownership'] = df['Curr_ownership'].fillna('פרטית')

    prev_data = {
        'Prev_ownership': [
            'פרטית', 'ליסינג', 'לא מוגדר', 'השכרה', 'אחר', 'חברה', 'מונית', None, 'ממשלתי'
        ],
        'Count': [543, 103, 43, 36, 27, 14, 4, 3, 1]
    }
    df_counts_prev = pd.DataFrame(prev_data)
    total_count_prev = df_counts_prev['Count'].sum()
    df_counts_prev['Probability'] = df_counts_prev['Count'] / total_count_prev

    nan_indices = df['Prev_ownership'][df['Prev_ownership'].isna()].index
    fill_values = random.choices(prev_data['Prev_ownership'], df_counts_prev['Probability'], k=len(nan_indices))
    df['Prev_ownership'].loc[nan_indices] = fill_values

    return df
