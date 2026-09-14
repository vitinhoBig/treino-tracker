from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# folder where the treino.py live - regardless of where the terminal is running from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

name = "Victor"
weight = 158.7
goal_days = 3
goal_sets = 15

print(f"Athlete: {name}")
print(f"Weight: {weight} lbs")
print(f"Goal: {goal_days} days/week, {goal_sets} sets/workout")


tuesday = {
    "day": "Tuesday",
    "focus": "Chest, shoulders and triceps",
    "wellness": {
        "sleep_hours": 8.0,
        "water_liters": 3.0,
        "rpe": 9,              # Rate of Perceived Exertion (1-10)
        "calories": 2205,
        "protein_g": 132
    },
    "exercises": [
        {"name": "Incline bench press", "sets": 3, "reps": 8, "load": 150},
        {"name": "Machine flat bench press", "sets": 3, "reps": 8, "load": 170},
        {"name": "Chest fly", "sets": 3, "reps": 10, "load": 115},
        {"name": "Cable lateral raises", "sets": 3, "reps": 12, "load": 30},
        {"name": "Cable front raises", "sets": 3, "reps": 8, "load": 60},
        {"name": "French press", "sets": 3, "reps": 10, "load": 120},
        {"name": "Triceps pushdown", "sets": 2, "reps": 10, "load": 140}

    ]
}

thursday = {
    "day": "Thursday",
    "focus": "Back and biceps",
    "wellness": {
        "sleep_hours": 8.0,
        "water_liters": 2.5,
        "rpe": 9, 
        "calories": 2320,
        "protein_g": 142
    },
    "exercises": [
        {"name": "Lat pulldown", "sets": 3, "reps": 10, "load": 175},
        {"name": "T-bar row", "sets": 3, "reps": 10, "load": 145},
        {"name": "Single arm row", "sets": 3, "reps": 10, "load": 150},
        {"name": "Cable pullover", "sets": 3, "reps": 10, "load": 120},
        {"name": "Incline bicep curl", "sets": 3, "reps": 10, "load": 120},
        {"name": "Hammer curl", "sets": 3, "reps": 10, "load": 130},

    
    
    ]
}

saturday = {
    "day": "Saturday",
    "focus": "Legs",
    "wellness": {
        "sleep_hours": 8.0,
        "water_liters": 3.0,
        "rpe": 10,
        "calories": 2395,
        "protein_g": 167

    },
    "exercises": [
        {"name": "Squats", "sets": 4, "reps": 10, "load": 250},
        {"name": "Leg extension", "sets": 3, "reps": 10, "load": 150},
        {"name": "Leg curl", "sets": 3, "reps": 10, "load": 135}, 
        {"name": "Calf raise", "sets": 3, "reps": 12, "load": 160},
        {"name": "Adductor machine", "sets": 3, "reps": 10, "load": 95}
    ]
}

def total_sets(day):
    """Counts the total sets in a workout"""
    total = 0
    for ex in day["exercises"]:
        total += ex["sets"]
    return total

def total_reps(day):
    """Counts the total reps in a workout"""
    total = 0
    for ex in day["exercises"]:
        total += ex["sets"] * ex["reps"]    # sets * reps
    return total

def total_volume(day):
    """Calculates sets x reps x load - the real workout metric"""
    total = 0
    for ex in day["exercises"]:
        total += ex["sets"] * ex["reps"] * ex["load"]  # sets x reps x load
    return total

def get_wellness_summary(week):
    """Averages wellness metric across the week's training days."""
    sleep    = [d["wellness"]["sleep_hours"] for d in week]
    water    = [d["wellness"]["water_liters"] for d in week]
    rpe      = [d["wellness"]["rpe"] for d in week]
    calories = [d["wellness"]["calories"] for d in week]
    protein  = [d["wellness"]["protein_g"] for d in week]

    return {
        "avg_sleep_hours":  round(np.mean(sleep), 1),
        "avg_water_liters": round(np.mean(water), 1),
        "avg_rpe":          round(np.mean(rpe), 1),
        "avg_calories":     round(np.mean(calories)),
        "avg_protein_g":    round(np.mean(protein))
    }

def create_dataframe(week):
    """Converts the training week into a pandas DataFrame."""
    rows = []
    for day in week:
        for ex in day["exercises"]:
            row = ex.copy()
            row["day"]  = day["day"]
            row["focus"] = day["focus"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df["volume"] = df["sets"] * df["reps"] * df["load"]
    return df

def classify_load(load):
    """Classifies a single exercise's intensity based on load - same theresholds as load_progression."""
    if load < 110:
        return "Light"
    elif load < 155:
        return "Moderate"
    elif load < 200:
        return "Heavy"
    else:
        return "Very heavy"
    
def train_intensity_classifier(week):
    """Trains a Random Forest to predict exercise intensity from sets, reps, and load."""""
    df = create_dataframe(week)
    df["intensity"] = df["load"].apply(classify_load)

    X = df[["sets", "reps", "load"]]
    y = df["intensity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"\n--- Intensity classifier ---")
    print(f"  Trained on {len(X_train)} exercises, tested on {len(X_test)}")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    
    return model

def predict_workout_intensity(model, sets, reps, load):
    """Predicts the intensity classification for a hypothetical new exercise."""
    input_df = pd.DataFrame([[sets, reps, load]], columns=["sets", "reps", "load"])
    prediction = model.predict(input_df)[0]
    print(f"  Predicted intensity for {sets}x{reps} @ {load}lbs: {prediction}")
    return prediction 

def diagnose_classifier(week):
    """Runs a full diagnostic on the intensity classifier using cross-validation, instead of trusting a single accuracy score."""
    df = create_dataframe(week)
    df["intensity"] = df["load"].apply(classify_load)

    X = df[["sets", "reps", "load"]]
    y = df["intensity"]

    print(f"\n--- Classifier diagnostic ---")
    print(f"  Dataset size: {len(df)} exercises")

    cv = min(5, len(df) // 3) 
    if cv < 2:
        print("  Not enough data for cross-validation yet.")
        return

    for depth in [2, 3, 5]:
        model = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv)
        print (f"  max_depth={depth}: {scores.mean()*100:.0f}% (+/- {scores.std()*100:.0f}%)")

def cluster_exercises(week, n_clusters=3):
    """Groups exercises into natural clusters using K-Means, based on sets, reps, and load."""
    df = create_dataframe(week)
    X = df[["sets", "reps", "load"]]

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)

    print(f"\n--- Exercise clusters ---")
    for c in sorted(df["cluster"].unique()):
        group = df[df["cluster"] == c]
        avg_load = group["load"].mean()
        print(f"  Cluster {c} ({len(group)} exercises) - avg load: {avg_load:.0f} lbs")
        for exercise_name in group["name"]:
            print(f"      - {exercise_name}")

    return df

def pandas_analysis(week, week_number):
    """Generates a full weekly analysis using Pandas and Numpy."""
    df = create_dataframe(week)

    print(f"\n================================")
    print(f"   Pandas analysis - Week {week_number}")
    print(f"================================")

    # volume per day
    print("\nVolume per day:")
    vol_day = df.groupby("day")["volume"].sum().sort_values(ascending=False)
    for day, vol in vol_day.items():
        print(f"  {day}: {vol} lbs")

    # heaviest exercise
    heaviest = df.sort_values("load", ascending=False).iloc[0]
    print(f"\nHeaviest exercise:")
    print(f"  {heaviest['name']} — {heaviest['load']} lbs ({heaviest['day']})")

    # exercise with highest volume
    highest_vol = df.sort_values("volume", ascending=False).iloc[0]
    print(f"\nHighest volume exercise:")
    print(f"  {highest_vol['name']} - {highest_vol['volume']} lbs")

    # load statistics with Numpy
    loads = np.array(df["load"])
    print(f"\nLoad statistics:")
    print(f"  Mean:            {np.mean(loads):.1f} lbs")
    print(f"  Highest load:    {np.max(loads)} lbs")
    print(f"  Lowest load:     {np.min(loads)} lbs")
    print(f"  Std deviation:   {np.std(loads):.1f} lbs")

    # top 3 exercises by volume
    print(f"\nTop 3 exercises by volume:")
    top3 = df.sort_values("volume", ascending=False).head(3)
    for _, row in top3.iterrows():
        print(f"  {row['name']}: {row['volume']} lbs ({row['day']})")

    # save CSV
    filename = f"workout_week{week_number}.csv"
    df.to_csv(get_path(filename), index=False)
    print(f"\nFile {filename} saved!")

    return df

def plot_weekly_volume(week, week_number, name):
    """Plots the total volume per day using matplotlib and saves as PNG."""
    df = create_dataframe(week)
    vol_day = df.groupby("day")["volume"].sum().sort_values(ascending=False)

    plt.figure(figsize=(7, 4))
    plt.bar(vol_day.index, vol_day.values, color="steelblue")
    plt.title(f"Total volume per day - {name} (Week {week_number})")
    plt.xlabel("Day")
    plt.ylabel("Volume (lbs)")
    plt.tight_layout()

    filename = f"volume_week{week_number}.png"
    plt.savefig(get_path(filename))
    plt.close()
    
def classify_workout(sets):
    """Classifies the workout by set volume"""
    if sets < 10:
        return "Light"
    elif sets < 15:
        return "Moderate"
    elif sets < 20:
        return "Heavy"
    else:
        return "Very heavy"

def hit_goal(day, goal):
    return total_sets(day) >= goal

def workout_summary(day, goal):
    """Prints the full summary of a workout day"""
    print(f"\n--- {day['day']} ({day['focus']}) ---")

    for ex in day["exercises"]:
        print(f" {ex['name']}: {ex['sets']}x{ex['reps']} @ {ex['load']}lbs")

    s = total_sets(day)
    classification = classify_workout(s)

    print(f"  Total sets: {s}")
    print(f"  Total reps: {total_reps(day)}")
    print(f"  Total volume: {total_volume(day)} lbs")

    if s >= goal:
        print(f"  Classification: {classification} - Goal hit!")
    elif s >= goal - 3:
        print(f"  Classification: {classification} - Almost! {goal - s} sets short.")
    else:
        print(f"  Classification: {classification} - {goal - s} sets short.")

def generate_load_progression(starting_load, goal_load, increment=5.0):
    """Generates a list of week/load/reps data for a progression, without printing."""
    weeks = []
    loads = []
    reps_list = []

    week = 1
    load = starting_load

    while load <= goal_load:
        if load < 110:
            reps = 12
        elif load < 155:
            reps = 10
        elif load < 200:
            reps = 8
        else:
            reps = 6
        
        weeks.append(week)
        loads.append(load)
        reps_list.append(reps)

        load += increment
        week += 1

    return weeks, loads, reps_list

def load_progression(exercise, starting_load, goal_load, increment=5.0):
    """Prints the weekly load progression until the goal is reached."""
    print(f"\n--- Load progression: {exercise} ---")
    
    weeks, loads, reps_list = generate_load_progression(starting_load, goal_load, increment)

    for w, l, r in zip(weeks, loads, reps_list):
        print(f"  Week {w:2}: {l:.1f} lbs x {r} reps")

    print(f" Goal of {goal_load:.1f} lbs reached!")

def plot_load_progression(exercise, starting_load, goal_load, increment=5.0):
    """Plots the weekly load progression as a line chart and saves as PNG."""
    weeks, loads, reps_list = generate_load_progression(starting_load, goal_load, increment)

    plt.figure(figsize=(7, 4))
    plt.plot(weeks, loads, marker="o", color="steelblue")
    plt.title(f"Load progression - {exercise}")
    plt.xlabel("Week")
    plt.ylabel("Load (lbs)")
    plt.tight_layout()

    safe_name = exercise.lower().replace(" ", "_")
    filename = f"progression_{safe_name}.png"
    plt.savefig(get_path(filename))
    plt.close()

    print(f"\nChart saved as {filename}!")


def weekly_report(name, workouts, goal_sets):
    """Generates the full weekly report."""
    print(f"\n================================")
    print(f"   Weekly report - {name}")
    print(f"================================")

    total_s  = []
    total_v  = []

    for day in workouts:
        workout_summary(day, goal_sets)
        total_s.append(total_sets(day))
        total_v.append(total_volume(day))

    print(f"\nAverage sets per workout: {round(sum(total_s) / len(total_s), 1)}")
    print(f"Total sets for the week:  {sum(total_s)}")
    print(f"Total volume for the week:    {sum(total_v)} lbs")

def save_history(name, week, week_number):
    """Saves the week's history to history.json"""

    wellness = get_wellness_summary(week)

    record = {
        "week":             week_number,
        "athlete":          name,
        "total_sets":       sum(total_sets(d) for d in week),
        "total_volume":     sum(total_volume(d) for d in week),
        "avg_sleep_hours":  wellness["avg_sleep_hours"],
        "avg_water_liters": wellness["avg_water_liters"],
        "avg_rpe":          wellness["avg_rpe"],
        "avg_calories":     wellness["avg_calories"],
        "avg_protein_g":    wellness["avg_protein_g"],
    }

    try:
        with open(get_path("history.json"), "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    # check if this week already exists in the history
    found = False
    for i, w in enumerate(history):
        if w["week"] == week_number:
            history[i] = record  # replace only the FIRST one found
            found = True
            break           # stop here, don't look at the others

    if not found:
        history.append(record)     # add only if it's new

    with open(get_path("history.json"), "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nWeek {week_number} saved to history.json!")

def view_history():
    """Loads and displays the weekly history."""
    try:
        with open(get_path("history.json"), "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("No history found yet.")
        return

    print("\n--- Workout history ---")
    for w in history:
        print(f"Week {w['week']}: {w['total_sets']} sets | {w['total_volume']} lbs | "
              f"sleep {w.get('avg_sleep_hours', '?')}h | RPE {w.get('avg_rpe', '?')}")

    if len(history) >= 2:
        diff = history[-1]["total_volume"] - history[-2]["total_volume"]
        sign = "+" if diff >= 0 else ""
        print(f"\n   Change vs previous week: {sign}{diff} lbs")

def predict_next_week():
    """Predicts the next week's total volume using a recency-weighted linear regression 
    over the most recent weeks (recent weeks count more), and flags nomalous weeks 
    using z-scores over the full history."""
    try:
        with open(get_path("history.json"), "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("No history found yet.")
        return
    
    if len(history) < 2:
        print("\nNo history found yet - can't predict.")
        return
    
    weeks_list = [w["week"] for w in history]
    volumes = np.array([w["total_volume"] for w in history])

    # anomaly detection using over the FULL history - useful context, even if not used for the fit
    mean = np.mean(volumes)
    std = np.std(volumes)
    z_scores = (volumes - mean) / std if std > 0 else np.zeros(len(volumes))
    anomalies = [weeks_list[i] for i in range(len(volumes)) if abs(z_scores[i]) > 1]

    print(f"\n--- Volume prediction ---")
    if anomalies:
        print(f"  Note: week(s) {anomalies} look unusual (more than 1 std from average) - trend may be skewed by them.")

    # fit only on the most recent weeks, using the RAW volumes (no rolling average) -
    # smoothing and recency-weighting were fighting each other)
    recent_n = min(5, len(history))
    recent_weeks = weeks_list[-recent_n:]
    recent_volumes = volumes[-recent_n:]

    weeks_for_fit = [[w] for w in recent_weeks]
    y_for_fit = recent_volumes.tolist()

    # recency weights: oldest week in the window = weight 1, most recent = weight recent_n
    sample_weights = list(range(1, recent_n + 1))

    model = LinearRegression()
    model.fit(weeks_for_fit, y_for_fit, sample_weight=sample_weights)
    next_week = weeks_list[-1] + 1
    prediction = model.predict([[next_week]])[0]
    trend = model.coef_[0]

    note = f" (last {recent_n} weeks, recency-weighted)"
    print(f"  Predicted volume for week {next_week}: {prediction:.1f} lbs{note}")
    print(f"  Weekly trend: {trend:+.1f} lbs/week")

    if trend > 100:
        print("  You're trending upward - keep it up!")
    elif trend < -100:
        print("  Volume is trending down - might be a good week to push harder.")
    else:
        print("  Volume is holding steady.")

def predict_with_wellness():
    """Compares a simple model (week only) vs an enriched model (+ sleep, RPE)."""
    try:
        with open(get_path("history.json"), "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("\nNo history found yet.")
        return

    #only use weeks that have complete wellness data
    complete = [w for w in history if "avg_sleep_hours" in w]

    if len(complete) < 4:
        print(f"\nOnly {len(complete)} weeks with wellness data.")
        print("Need at least 4 for a meaningful comparison - keep logging!")
        return

    X_simple = [[w["week"]] for w in complete]
    X_rich   = [[w["week"], w["avg_sleep_hours"], w["avg_rpe"]] for w in complete]
    y = [w["total_volume"] for w in complete]

    model_simple = LinearRegression().fit(X_simple, y)
    model_rich   = LinearRegression().fit(X_rich, y)

    print(f"\n --- Feature comparison ({len(complete)} weeks with wellness data) ---")
    print(f"  Simple (week only):     R2 = {model_simple.score(X_simple, y):.2f}")
    print(f"  Enriched (+sleep, RPE): R2 = {model_rich.score(X_rich, y):.2f}")
    print(f"  (More weeks needed before trusting which model is really better)")

# call the report — last lines of the file
week = [tuesday, thursday, saturday]
weekly_report(name, week, goal_sets)

save_history(name, week, week_number=11)
view_history()

predict_next_week()

predict_with_wellness()  

intensity_model = train_intensity_classifier(week)
predict_workout_intensity(intensity_model, sets=4, reps=8, load=180) #example prediction
predict_workout_intensity(intensity_model, sets=3, reps=12, load=30) #another example prediction

diagnose_classifier(week)

clustered_df = cluster_exercises(week, n_clusters=3)

df = pandas_analysis(week, week_number=11)

plot_weekly_volume(week, week_number=11, name=name)

print("\n")
load_progression("Squat", starting_load=130, goal_load=220, increment=5.0)
plot_load_progression("Squat", starting_load =130, goal_load=220, increment=5.0)

load_progression("Flat bench press", starting_load=155, goal_load=200, increment=5.0)
plot_load_progression("Flat bench press", starting_load=155, goal_load=200, increment=5.0)


