from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import os
import re

class PlotResults:
    def __init__(self, scores, times, permutations, question):
        self.scores = scores
        self.times = times
        self.permutations = permutations
        self.question = question

    def plot_results(self):
        # Assign a unique color for each permutation
        cmap = get_cmap('tab10')
        colors = [cmap(i) for i in range(len(self.permutations))]
        perm_color_map = {perm: colors[i] for i, perm in enumerate(self.permutations)}

        # Create scatter plot
        plt.figure(figsize=(16, 5))
        for i, perm in enumerate(self.permutations):
            plt.scatter(self.times[i], self.scores[i], color=perm_color_map[perm], label=f'LLM: {perm[0]} | Vision LLM: {perm[1]}')

        plt.title(f'Scores vs Time for Question: {self.question}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Score')
        plt.legend(title='Permutation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        question = re.sub(r'[^a-zA-Z0-9 ]', '', self.question)
        plt.savefig(f"{os.getenv('PLOT_SAVE_DIR')}\\question_{question}.png")
