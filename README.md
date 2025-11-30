# Medical_ChatBot
Developing a medical chatbot with RAG and LLMs

## Dataset: KERAAL (KinesiothErapy and Rehabilitation for Assisted Ambient Living)

The KERAAL Dataset is a medical database of clinical patients carrying out low back-pain rehabilitation exercises. It can be used as a data source for training and evaluating medical chatbots focused on physical rehabilitation.

### Dataset Overview

The KERAAL dataset is designed for human body movement analysis in the context of low-back pain physical rehabilitation. This dataset was acquired during a clinical study where patients performed 3 low-back pain rehabilitation exercises while being coached by a robotic system.

**Key Features:**
- Recorded from 9 healthy subjects and 12 patients suffering from low-back pain
- Annotated by two rehabilitation doctors
- Includes 3D skeleton sequences captured by Kinect
- RGB videos and 2D skeleton data estimated from videos
- Medical expert annotations for:
  - Assessment of correctness
  - Recognition of errors
  - Spatio-temporal localization of errors

### Dataset Content

| Group | Annotation | RGB Videos | Kinect | Openpose/Blazepose | Vicon | Nb Recordings |
|-------|-----------|------------|--------|-------------------|-------|---------------|
| 1a | xml anvil: err label, bodypart, timespan | mp4, 480x360 | tabular | dictionary | NA | 249 |
| 1b | NA | mp4, 480x360 | tabular | dictionary | NA | 1631 |
| 2a | xml anvil: err label, bodypart, timespan | mp4, 480x360 | tabular | dictionary | NA | 51 |
| 2b | NA | mp4, 480x360 | tabular | dictionary | NA | 151 |
| 3 | error label | avi, 960x544 | tabular | dictionary | tabular | 540 |

### Data Types Included

- **RGB Videos**: Anonymized videos in mp4/avi format
- **Kinect Skeleton Data**: 3D positions and orientations of joints in tabular ASCII format
- **OpenPose/BlazePose Skeleton Data**: 2D/3D positions in COCO pose format
- **Vicon Motion Capture Data**: High-precision skeleton sequences
- **Annotations**: XML anvil format with error labels, body parts, and temporal descriptions

### Links and Resources

- **GitHub Repository**: [https://github.com/nguyensmai/KeraalDataset](https://github.com/nguyensmai/KeraalDataset)
- **Dataset Website**: [https://keraal.enstb.org/KeraalDataset.html](https://keraal.enstb.org/KeraalDataset.html)
- **Download**: [http://nguyensmai.free.fr/KeraalDataset.html](http://nguyensmai.free.fr/KeraalDataset.html)

### Citation

If using this dataset, please cite:

```bibtex
@inproceedings{Nguyen2024IJCNN,
    author = {Sao Mai Nguyen and Maxime Devanne and Olivier Remy-Neris and Mathieu Lempereur and Andre Thepaut},
    booktitle = {International Joint Conference on Neural Networks},
    title = {A Medical Low-Back Pain Physical Rehabilitation Database for Human Body Movement Analysis},
    year = {2024}
}
```

### Use Cases for Medical ChatBot

This dataset can be used for:
1. **Exercise Assessment**: Training models to evaluate if rehabilitation exercises are performed correctly
2. **Error Recognition**: Identifying common mistakes patients make during exercises
3. **Feedback Generation**: Generating personalized feedback for patients based on their performance
4. **Rehabilitation Guidance**: Providing context-aware recommendations for low-back pain rehabilitation

---

## Projet Académique: KineIA - Assistant de Rééducation

Cette section propose une architecture simple et accessible pour développer une application "KineIA" basée sur le dataset KERAAL.

### Architecture Proposée

```
┌─────────────────────────────────────────────────────────────┐
│                    Application KineIA                        │
├─────────────────────────────────────────────────────────────┤
│  1. CAPTURE          │  2. ANALYSE           │  3. FEEDBACK │
│  ─────────────────   │  ─────────────────    │  ──────────  │
│  - Webcam            │  - Extraction pose    │  - Correct/  │
│  - Vidéo uploadée    │  - Classification     │    Incorrect │
│                      │    d'erreurs          │  - Type      │
│                      │                       │    d'erreur  │
│                      │                       │  - Conseils  │
└─────────────────────────────────────────────────────────────┘
```

### Étapes de Développement (Niveau Académique)

#### Étape 1: Préparation des Données
```python
# Charger les données de squelette Kinect (format tabulaire)
import pandas as pd

# Les fichiers .txt contiennent les positions des joints
# Format: timestamp, joint1_x, joint1_y, joint1_z, ...
skeleton_data = pd.read_csv('skeleton_file.txt', delimiter=' ')
```

#### Étape 2: Extraction de Caractéristiques
```python
# Calculer des angles entre les joints (features simples)
import numpy as np

def calculate_angle(joint1, joint2, joint3):
    """Calcule l'angle entre trois points (joints)"""
    v1 = joint1 - joint2
    v2 = joint3 - joint2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1))

# Exemple: angle du dos (épaule-hanche-genou)
```

#### Étape 3: Classification Simple
```python
# Utiliser scikit-learn pour un classificateur accessible
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# X = features extraites (angles, distances)
# y = labels (correct=0, erreur_type1=1, erreur_type2=2, ...)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

#### Étape 4: Interface Utilisateur (Flask)
```python
from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Recevoir la vidéo ou image
    # Extraire le squelette avec MediaPipe
    # Classifier le mouvement
    # Retourner le feedback
    pass
```

### Technologies Recommandées (Accessibles)

| Composant | Technologie | Niveau de Difficulté |
|-----------|-------------|---------------------|
| Extraction de pose | MediaPipe (Python) | ⭐ Facile |
| Preprocessing | Pandas, NumPy | ⭐ Facile |
| Classification | Scikit-learn | ⭐⭐ Moyen |
| Interface web | Flask | ⭐⭐ Moyen |
| Visualisation | Matplotlib, OpenCV | ⭐ Facile |

### Structure de Projet Suggérée

```
KineIA/
├── data/
│   ├── raw/              # Données KERAAL brutes
│   └── processed/        # Données prétraitées
├── src/
│   ├── preprocessing.py  # Chargement et nettoyage
│   ├── features.py       # Extraction de caractéristiques
│   ├── model.py          # Entraînement du modèle
│   └── inference.py      # Prédiction en temps réel
├── app/
│   ├── app.py            # Application Flask
│   └── templates/        # Pages HTML
├── notebooks/
│   └── exploration.ipynb # Analyse exploratoire
├── requirements.txt
└── README.md
```

### Objectifs d'Apprentissage

Ce projet permet de maîtriser:
- 📊 **Traitement de données**: Manipulation de séries temporelles de squelettes
- 🤖 **Machine Learning**: Classification supervisée avec features manuelles
- 🎥 **Vision par ordinateur**: Détection de pose avec MediaPipe
- 🌐 **Développement web**: Création d'une interface avec Flask
- 📈 **Évaluation**: Métriques de performance (accuracy, precision, recall)

### Ressources pour Débuter

1. **MediaPipe Pose**: [Documentation officielle](https://google.github.io/mediapipe/solutions/pose.html)
2. **Scikit-learn**: [Tutoriels de classification](https://scikit-learn.org/stable/supervised_learning.html)
3. **Code KERAAL**: Le repository inclut des exemples LSTM et GMM dans `evaluation_mouvements_lstm_python/`
