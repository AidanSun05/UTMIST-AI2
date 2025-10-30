import glob
import os
import re
import sys

from environment.agent import run_match
from environment.environment import Result
from user.my_agent import SubmittedAgent

MODELS_DIR = "checkpoints/experiment_9"
MODELS = sorted(glob.glob(os.path.join(MODELS_DIR, "*.zip")))
STEPS_PATTERN = re.compile(r"rl_model_(.+)_steps\.zip")

best_model = MODELS[0]

for next_model in MODELS[1:]:
    match1 = STEPS_PATTERN.search(best_model)
    match2 = STEPS_PATTERN.search(next_model)
    if not match1 or not match2:
        continue

    steps1 = match1.group(1)
    steps2 = match2.group(1)

    print(f"========== {steps1}-{steps2} ==========")

    best_wins = 0
    next_wins = 0
    for i in range(50):
        print(f"{i + 1}/50")
        best_agent = SubmittedAgent(file_path=best_model)
        next_agent = SubmittedAgent(file_path=next_model)

        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")

        stats = run_match(best_agent, next_agent, max_timesteps=9999)
        if stats.player1_result == Result.WIN:
            best_wins += 1
        elif stats.player1_result == Result.LOSS:
            next_wins += 1

        sys.stdout = old_stdout

    print(f"========== {steps1}: {best_wins}, {steps2}: {next_wins} ==========")
    if next_wins >= best_wins:
        best_model = next_model

print(f"========== Best model: {best_model} ==========")
