# Treino Tracker

A simple command-line workout tracker written in Python. It logs weekly training sessions (sets, reps, and load per exercise), calculates training volume, classifies workout intensity, tracks load progression toward a goal, and persists weekly history to a JSON file.

## Features

- Define weekly workout days with exercises, sets, reps, and load (in lbs)
- Calculate total sets, total reps, and total training volume per workout
- Classify each workout as Light, Moderate, Heavy, or Very heavy based on set volume
- Generate a full weekly report across all workout days
- Track weekly load progression toward a target goal (e.g. squat, bench press)
- Save and update weekly history in `history.json`, with automatic week overwrite if re-run
- View historical progress, including volume change vs. the previous week

## How to run

```bash
python treino.py
```

## Example output

```
Athlete: Victor
Weight: 175.0 lbs
Goal: 3 days/week, 15 sets/workout

================================
   Weekly report - Victor
================================

--- Tuesday (Chest, shoulders and triceps) ---
 Incline bench press: 3x10 @ 150lbs
 Flat bench press: 2x8 @ 165lbs
 ...
  Total sets: 19
  Total reps: 192
  Total volume: 22220 lbs
  Classification: Heavy - Goal hit!

Average sets per workout: 17.3
Total sets for the week:  52
Total volume for the week:    72680 lbs

Week 1 saved to history.json!
```

## Roadmap

- Migrate workout and history data structures to Pandas DataFrames for easier analysis and filtering
- Add data visualization (e.g. volume trends over time with Matplotlib)
- Support multiple athletes / profiles

## Tech stack

- Python 3
- `json` (standard library) for history persistence
- Pandas / NumPy (in progress — planned migration, see Roadmap)
