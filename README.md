# 🏋️ Treino Tracker

A personal workout tracking system built from scratch in Python — combining real training data with data analysis and machine learning to understand, visualize, and predict my own progress in the gym.

This started as a beginner Python exercise (variables, lists, functions) and grew, week by week, into a full data pipeline as I learned Pandas, NumPy, Matplotlib, and Scikit-learn. Every feature was added to solve a real problem I ran into tracking my own training — not as a disconnected tutorial exercise.

---

## What it does

Each week, I log my workouts (sets, reps, load, wellness) as structured Python data. Running the script produces:

- **A full workout report** — per-exercise breakdown, volume, and a Light/Moderate/Heavy/Very Heavy classification per day
- **A persistent history** (`history.json`) tracking every week — sets, volume, and wellness metrics, without duplicating entries
- **Pandas/NumPy analysis** — volume per day, heaviest exercise, load statistics, top exercises by volume
- **Matplotlib charts** — weekly volume bar charts and per-exercise load progression line charts, auto-saved as PNGs
- **Wellness tracking** — sleep, water intake, RPE (Rate of Perceived Exertion), calories, and protein per day
- **Machine learning:**
  - **Linear Regression** — predicts next week's total volume from historical trend
  - **Random Forest Classifier** — predicts exercise intensity from sets/reps/load
  - **K-Means Clustering** — discovers natural exercise groupings with no labels provided
  - **Cross-validation diagnostics** — evaluates the classifier honestly instead of trusting a single train/test split
  - **Feature engineering comparison** — tests whether adding sleep and RPE actually improves volume predictions, and refuses to draw conclusions until there's enough data to trust them

---

## Why this approach

Real personal data is messy and small. Some weeks are strong, some are recovery weeks from being sick, some skip an exercise entirely. This project treats that as a design constraint, not an inconvenience:

- Functions handle missing wellness data from earlier weeks gracefully (`.get()` with fallbacks)
- The feature-comparison model **refuses to run** until there are at least 4 weeks of complete data, and still flags the result as unreliable with so few points
- The classifier diagnostic uses cross-validation specifically because a single accuracy score misled me early on (a "100% accuracy" result in week 3 turned out to be a fluke of a tiny test set, not a good model)
- Outliers (like a heavy compound lift such as Squat) are left as outliers in clustering rather than forced into a "clean" group — the model reports what's actually there

---

## Tech stack

- **Python 3.9**
- **Pandas** & **NumPy** — data structuring and statistics
- **Matplotlib** — visualization
- **Scikit-learn** — `LinearRegression`, `RandomForestClassifier`, `KMeans`, `train_test_split`, `cross_val_score`
- **JSON** — persistent weekly history

---

## Project structure

```
treino-tracker/
├── treino.py                          # main script
├── history.json                       # weekly history (gitignored — personal data)
├── workout_week{N}.csv                # per-week exercise data export
├── volume_week{N}.png                 # weekly volume bar chart
├── progression_{exercise}.png         # load progression line chart per exercise
└── README.md
```

---

## How to run it

```bash
pip3 install pandas numpy matplotlib scikit-learn
python3 treino.py
```

Each week, I update the exercise dictionaries (`tuesday`, `thursday`, `saturday`) with that week's actual sets/reps/load and wellness data, bump the `week_number`, and run the script. The history accumulates automatically.

---

## Sample output

```
--- Workout history ---
Week 5: 52 sets | 67300 lbs | sleep 7.8h | RPE 9.0
Week 6: 52 sets | 66450 lbs | sleep 7.8h | RPE 9.0
Week 7: 49 sets | 60170 lbs | sleep 7.8h | RPE 8.7

--- Volume prediction ---
  Predicted volume for week 8: 59160.0 lbs
  Weekly trend: -1658.6 lbs/week

--- Classifier diagnostic ---
  Dataset size: 17 exercises
  max_depth=2: 77% (+/- 12%)
  max_depth=3: 90% (+/- 12%)
```

---

## What I learned building this

- Structuring real-world data as nested dictionaries and converting it into tidy DataFrames for analysis
- The difference between a single accuracy score and a trustworthy evaluation (cross-validation, train/test comparison)
- Why more features and a higher R² isn't automatically a better model — especially with small datasets
- Debugging real bugs: path resolution across working directories, variable name collisions, missing dictionary keys, and off-by-one logic in loops
- Git/GitHub workflow with meaningful, incremental commit history

---

## Roadmap

- Accumulate enough weeks for the wellness feature comparison to become statistically meaningful
- Expand the exercise-level dataset across multiple weeks for more robust classification
- Possibly explore a simple web dashboard once the CLI output outgrows the terminal
