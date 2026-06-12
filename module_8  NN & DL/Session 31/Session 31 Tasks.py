{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "219e07ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.5\n",
      "Model saved as music_genre_model.pkl\n",
      "Predicted Genre: Pop\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\skadi\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:2691: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# 1 \n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Sample dataset\n",
    "data = {\n",
    "    'tempo': [120, 140, 70, 130, 150, 80, 110, 160],\n",
    "    'energy': [0.8, 0.9, 0.3, 0.85, 0.95, 0.2, 0.7, 0.98],\n",
    "    'danceability': [0.9, 0.8, 0.4, 0.85, 0.75, 0.3, 0.8, 0.7],\n",
    "    'genre': ['Pop', 'Rock', 'Classical', 'Pop',\n",
    "              'Rock', 'Classical', 'Pop', 'Rock']\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "\n",
    "X = df[['tempo', 'energy', 'danceability']]\n",
    "y = df['genre']\n",
    "\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "y_pred = model.predict(X_test)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "\n",
    "\n",
    "joblib.dump(model, \"music_genre_model.pkl\")\n",
    "print(\"Model saved as music_genre_model.pkl\")\n",
    "\n",
    "\n",
    "loaded_model = joblib.load(\"music_genre_model.pkl\")\n",
    "new_song = [[125, 0.85, 0.88]]  \n",
    "prediction = loaded_model.predict(new_song)\n",
    "\n",
    "print(\"Predicted Genre:\", prediction[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92aef3d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Genre: Pop\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\skadi\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:2691: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# 2\n",
    "import joblib\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "\n",
    "model = joblib.load(\"music_genre_model.pkl\")\n",
    "\n",
    "X_train = np.array([\n",
    "    [120, 0.8, 0.9],\n",
    "    [140, 0.9, 0.8],\n",
    "    [70, 0.3, 0.4],\n",
    "    [130, 0.85, 0.85],\n",
    "    [150, 0.95, 0.75],\n",
    "    [80, 0.2, 0.3]\n",
    "])\n",
    "\n",
    "\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "\n",
    "\n",
    "new_song = np.array([[125, 0.85, 0.88]])\n",
    "new_song_scaled = scaler.transform(new_song)\n",
    "prediction = model.predict(new_song_scaled)\n",
    "\n",
    "print(\"Predicted Genre:\", prediction[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "629b1a34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['music_genre_model.pkl']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "joblib.dump(model, \"music_genre_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ff51c4e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Genre: Pop\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\skadi\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\sklearn\\utils\\validation.py:2691: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "model = joblib.load(\"music_genre_model.pkl\")\n",
    "scaler = joblib.load(\"scaler.pkl\")\n",
    "\n",
    "new_song = [[125, 0.85, 0.88]]\n",
    "\n",
    "new_song_scaled = scaler.transform(new_song)\n",
    "\n",
    "genre = model.predict(new_song_scaled)\n",
    "\n",
    "print(\"Predicted Genre:\", genre[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "30340a70",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-06-12 13:31:12.850 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.046 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\skadi\\AppData\\Roaming\\Python\\Python313\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-06-12 13:31:13.047 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.048 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.048 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-06-12 13:31:13.051 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# 3 \n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load model\n",
    "model = joblib.load(\"music_genre_model.pkl\")\n",
    "\n",
    "# Load scaler if used during training\n",
    "# scaler = joblib.load(\"scaler.pkl\")\n",
    "\n",
    "st.title(\"🎵 Music Genre Prediction\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\n",
    "    \"Upload CSV File\",\n",
    "    type=[\"csv\"]\n",
    ")\n",
    "\n",
    "if uploaded_file is not None:\n",
    "\n",
    "    # Read CSV\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "\n",
    "    st.subheader(\"Uploaded Data\")\n",
    "    st.dataframe(df)\n",
    "\n",
    "    # Apply preprocessing if used during training\n",
    "    # df_scaled = scaler.transform(df)\n",
    "\n",
    "    # Predict genres\n",
    "    predictions = model.predict(df)\n",
    "\n",
    "    # Add predictions to table\n",
    "    result_df = df.copy()\n",
    "    result_df[\"Predicted_Genre\"] = predictions\n",
    "\n",
    "    st.subheader(\"Prediction Results\")\n",
    "    st.dataframe(result_df)\n",
    "\n",
    "    # Download predictions\n",
    "    csv = result_df.to_csv(index=False).encode(\"utf-8\")\n",
    "\n",
    "    st.download_button(\n",
    "        \"Download Predictions\",\n",
    "        csv,\n",
    "        \"genre_predictions.csv\",\n",
    "        \"text/csv\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cf3df4a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Saved Successfully!\n"
     ]
    }
   ],
   "source": [
    "# 4\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Dummy Dataset\n",
    "data = {\n",
    "    \"tempo\":[120,125,130,140,150,145,70,75,80,118,135,72,122,148,78],\n",
    "    \"energy\":[0.85,0.80,0.78,0.95,0.98,0.92,0.20,0.25,0.30,0.82,0.90,0.22,0.84,0.96,0.28],\n",
    "    \"danceability\":[0.90,0.88,0.85,0.75,0.70,0.72,0.30,0.35,0.40,0.87,0.78,0.32,0.89,0.68,0.38],\n",
    "    \"genre\":[\"Pop\",\"Pop\",\"Pop\",\"Rock\",\"Rock\",\"Rock\",\n",
    "             \"Classical\",\"Classical\",\"Classical\",\n",
    "             \"Pop\",\"Rock\",\"Classical\",\"Pop\",\"Rock\",\"Classical\"]\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "X = df[[\"tempo\",\"energy\",\"danceability\"]]\n",
    "y = df[\"genre\"]\n",
    "\n",
    "# Scale Data\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Train/Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_scaled, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "# Train Model\n",
    "model = RandomForestClassifier(n_estimators=100)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save Model and Scaler\n",
    "joblib.dump(model, \"music_genre_model.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "\n",
    "print(\"Model Saved Successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15f1c959",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f06b163",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
