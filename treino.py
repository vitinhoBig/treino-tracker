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
weight = 175.0
goal_days = 3
goal_sets = 15

print(f"Athlete: {name}")
print(f"Weight: {weight} lbs")
print(f"Goal: {goal_days} days/week, {goal_sets} sets/workout")


tuesday = {
    "day": "Tuesday",
    "focus": "Chest, shoulders and triceps",
    "exercises": [
        {"name": "Incline bench press", "sets": 3, "reps": 8, "load": 165},
        {"name": "Flat bench press", "sets": 2, "reps": 8, "load": 175},
        {"name": "Chest fly", "sets": 3, "reps": 10, "load": 115},
        {"name": "Lateral raise", "sets": 3, "reps": 12, "load": 30},
        {"name": "Shoulder press", "sets": 3, "reps": 10, "load": 155},
        {"name": "French press", "sets": 3, "reps": 10, "load": 120},
        {"name": "Triceps rope pushdown", "sets": 2, "reps": 10, "load": 130}
    ]

}

thursday = {
    "day": "Thursday",
    "focus": "Back and biceps",
    "exercises": [
        {"name": "Lat pulldown", "sets": 3, "reps": 10, "load": 175},
        {"name": "Bent-over row", "sets": 3, "reps": 10, "load": 130},
        {"name": "Single-arm row", "sets": 3, "reps": 10, "load": 140},
        {"name": "Cable pullover", "sets": 2, "reps": 10, "load": 115},
        {"name": "Barbell curl", "sets": 3, "reps": 10, "load": 130},
        {"name": "Hammer curl", "sets": 3, "reps": 10, "load": 120},
        {"name": "Forearm curl", "sets": 2, "reps": 10, "load": 120}
    ]
}

saturday = {
    "day": "Saturday",
    "focus": "Legs",
    "exercises": [
        {"name": "Squat", "sets": 4, "reps": 10, "load": 265},
        {"name": "Leg extension", "sets": 3, "reps": 10, "load": 140},
        {"name": "Leg curl", "sets": 3, "reps": 10, "load": 120},
        {"name": "Calf raise", "sets": 4, "reps": 12, "load": 150}
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
    plt.show()
    
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

def load_progression(exercise, starting_load, goal_load, increment=5.0):
    """Shows the weekly load progression until the goal is reached."""
    print(f"\n--- Load progression: {exercise} ---")
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

        print(f"  Week {week:2}: {load:.1f} lbs x {reps} reps")
        load += increment
        week += 1
    print(f" Goal of {goal_load:.1f} lbs reached!")

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

    record = {
        "week":          week_number,
        "athlete":       name,
        "total_sets":    sum(total_sets(d) for d in week),
        "total_volume":  sum(total_volume(d) for d in week),

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
        print(f"Week {w['week']}: {w['total_sets']} sets | {w['total_volume']} lbs")

    if len(history) >= 2:
        diff = history[-1]["total_volume"] - history[-2]["total_volume"]
        sign = "+" if diff >= 0 else ""
        print(f"\n   Change vs previous week: {sign}{diff} lbs")


# call the report — last lines of the file
week = [tuesday, thursday, saturday]
weekly_report(name, week, goal_sets)

save_history(name, week, week_number=2)
view_history()

df = pandas_analysis(week, week_number=2)

plot_weekly_volume(week, week_number=2, name=name)

print("\n")
load_progression("Squat", starting_load=130, goal_load=220, increment=5.0)
load_progression("Flat bench press", starting_load=155, goal_load=200, increment=5.0)
