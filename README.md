# Spam Detection App

A simple spam detection project using **Logistic Regression** with `scikit-learn`.  
This project can classify messages as spam or ham (not spam) and includes a Flask app to test predictions.

---

## Project Files

- `model.py` : Script to train the Logistic Regression model on the dataset and save it as `model.pkl`.
- `app.py` : Flask web application to load the trained model and provide an interface to predict messages.
- `spam.csv` : Dataset of spam and ham messages used for training.
- `requirements.txt` : Python dependencies for the project.
- `.gitignore` : Git ignore file (includes `model.pkl`).
- `README.md` : This file.

---

## Installation

1. Clone the repository:

````bash
git clone https://github.com/your-username/your-repo.git
cd your-repo



2. Create a virtual environment:

```bash
python3 -m venv venv
````

3.Activate the virtual environment:

```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

4.Install dependencies:

```bash
pip install -r requirements.txt
```
