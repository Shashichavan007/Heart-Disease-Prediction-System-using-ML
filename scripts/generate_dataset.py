import csv
import random

# Set fixed seed for reproducibility
random.seed(42)

header = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

rows = []

# Generate 303 realistic patient records adhering to Cleveland Heart Disease Dataset statistics
for i in range(303):
    # Determine target (0: No heart disease risk, 1: Heart disease risk)
    # roughly 54% positive, 46% negative in Cleveland dataset
    target = 1 if (i % 2 == 0 or random.random() < 0.52) else 0

    if target == 1:
        age = int(random.gauss(56, 8))
        sex = 1 if random.random() < 0.72 else 0
        cp = random.choices([0, 1, 2, 3], weights=[0.45, 0.20, 0.25, 0.10])[0]
        trestbps = int(random.gauss(136, 18))
        chol = int(random.gauss(252, 48))
        fbs = 1 if random.random() < 0.18 else 0
        restecg = random.choices([0, 1, 2], weights=[0.45, 0.50, 0.05])[0]
        thalach = int(random.gauss(138, 22))
        exang = 1 if random.random() < 0.55 else 0
        oldpeak = round(max(0.0, random.gauss(1.6, 1.2)), 1)
        slope = random.choices([0, 1, 2], weights=[0.15, 0.60, 0.25])[0]
        ca = random.choices([0, 1, 2, 3, 4], weights=[0.35, 0.30, 0.20, 0.10, 0.05])[0]
        thal = random.choices([0, 1, 2, 3], weights=[0.05, 0.15, 0.65, 0.15])[0]
    else:
        age = int(random.gauss(52, 9))
        sex = 1 if random.random() < 0.55 else 0
        cp = random.choices([0, 1, 2, 3], weights=[0.15, 0.25, 0.40, 0.20])[0]
        trestbps = int(random.gauss(128, 15))
        chol = int(random.gauss(238, 42))
        fbs = 1 if random.random() < 0.12 else 0
        restecg = random.choices([0, 1, 2], weights=[0.60, 0.38, 0.02])[0]
        thalach = int(random.gauss(158, 18))
        exang = 1 if random.random() < 0.15 else 0
        oldpeak = round(max(0.0, random.gauss(0.6, 0.8)), 1)
        slope = random.choices([0, 1, 2], weights=[0.45, 0.45, 0.10])[0]
        ca = random.choices([0, 1, 2, 3, 4], weights=[0.75, 0.15, 0.07, 0.02, 0.01])[0]
        thal = random.choices([0, 1, 2, 3], weights=[0.02, 0.80, 0.15, 0.03])[0]

    # Bounds clipping
    age = max(29, min(77, age))
    trestbps = max(94, min(200, trestbps))
    chol = max(126, min(564, chol))
    thalach = max(71, min(202, thalach))
    oldpeak = max(0.0, min(6.2, oldpeak))

    rows.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target])

with open("data/heart_sample.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Generated {len(rows)} records in data/heart_sample.csv")
