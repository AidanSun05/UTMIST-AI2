import glob
import os
import re
import sys

from environment.agent import run_match
from environment.environment import Result
from user.my_agent import SubmittedAgent

MODELS_DIR = "g"
NUM_MATCHES = 50
MATCH_TIMESTEPS = 9999

STEPS_PATTERN = re.compile(r"rl_model_([0-9]+)_steps\.zip")
MODELS = sorted([int(match.group(1)) for model in glob.glob(os.path.join(MODELS_DIR, "*.zip")) if (match := STEPS_PATTERN.search(model))])

print(f"========== Found {len(MODELS)} models. ==========")
best_model = MODELS[0]

for next_model in MODELS[1:]:
    print(f"========== {best_model}-{next_model} ==========")

    best_wins = 0
    next_wins = 0
    for i in range(NUM_MATCHES):
        print(f"{i + 1}/{NUM_MATCHES}")
        best_agent = SubmittedAgent(file_path=os.path.join(MODELS_DIR, f"rl_model_{best_model}_steps.zip"))
        next_agent = SubmittedAgent(file_path=os.path.join(MODELS_DIR, f"rl_model_{next_model}_steps.zip"))

        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")

        stats = run_match(best_agent, next_agent, max_timesteps=MATCH_TIMESTEPS)
        if stats.player1_result == Result.WIN:
            best_wins += 1
        elif stats.player1_result == Result.LOSS:
            next_wins += 1

        sys.stdout = old_stdout

    print(f"========== {best_model}: {best_wins} wins, {next_model}: {next_wins} wins ==========")
    if next_wins >= best_wins:
        best_model = next_model

print(f"========== Best model: {best_model} ==========")
