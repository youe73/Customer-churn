import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

import joblib
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

df_train = pd.read_csv("./src/customer_churn_dataset-training-master.csv")
df_test =  pd.read_csv("./src/customer_churn_dataset-testing-master.csv")

mergeddf = pd.concat([df_train, df_test], ignore_index=True, sort=False)

mergeddf_new = mergeddf.dropna()

def cleaning_transform(df):

    df["Age"] = df["Age"].astype(int)
    df["Tenure"] = df["Tenure"].astype(int)
    df["Usage Frequency"] = df["Usage Frequency"].astype(int)
    df["Support Calls"] = df["Support Calls"].astype(int)
    df["Payment Delay"] = df["Payment Delay"].astype(int)
    df["Last Interaction"] = df["Last Interaction"].astype(int)
    df["Churn"] = df["Churn"].astype(int)

    subscription_encoder = OrdinalEncoder(categories=[['Basic','Standard','Premium']])
    subtype = subscription_encoder.fit_transform(df[["Subscription Type"]])
    
    df["Subscription Type"] = subtype.astype(int)
    contract_encoder = OrdinalEncoder(categories=[['Annual', 'Quarterly', 'Monthly']])
    contract = contract_encoder.fit_transform(df[["Contract Length"]])    
    df["Contract Length"] =contract.astype(int)

    gender_encoded = pd.get_dummies(df["Gender"],prefix="Gen", dtype=int)
    df = pd.concat([df, gender_encoded], axis=1)
    return df


df_train_cleaned = cleaning_transform(mergeddf_new)

X = df_train_cleaned[['Age','Support Calls', 'Payment Delay','Tenure', 'Usage Frequency','Subscription Type',
       'Contract Length', 'Total Spend', 'Last Interaction',
       'Gen_Female', 'Gen_Male']]

y = df_train_cleaned["Churn"]

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# defining nn model
def build_nn_model(learningrate=0.001):
    nn_model = Sequential([
    Dense(64, input_shape=(11,), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
    ])

    nn_model.compile(
        optimizer=Adam(learning_rate=learningrate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return nn_model

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

keras_clf = KerasClassifier(
    build_fn=lambda: build_nn_model(),
    epochs=30,
    batch_size=256,
    verbose=1,
    callbacks=[early_stop],
    validation_split=0.2,
)

# pipeline to handling scaling and imbalance labels
pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('smote', SMOTE(random_state=42)),
    ('nn', keras_clf)
])

# train model
model = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Keras Churn Model")
plt.show()

# display classification report
print(classification_report(y_test, y_pred, digits=4))

history = pipeline.named_steps["nn"].model_.history

# save model and pipeline steps
joblib.dump(pipeline.named_steps["scaler"], "model/scaler.joblib")
joblib.dump(pipeline.named_steps["smote"], "model/smote.joblib") 

model = pipeline.named_steps["nn"].model_
model.save("model/customer_churn_model.keras")

def plot_history(hist):
    plt.figure(figsize=(14, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    

    plt.tight_layout()
    plt.show()

plot_history(history)
