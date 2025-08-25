from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

class PlotResults:
    def __init__(self, scores, times, permutations):
        self.scores = scores
        self.times = times
        self.permutations = permutations

    def plot_results(self):
        # Assign a unique color for each permutation
        cmap = get_cmap('tab10')
        colors = [cmap(i) for i in range(len(self.permutations))]
        perm_color_map = {perm: colors[i] for i, perm in enumerate(self.permutations)}

        # Create scatter plot
        plt.figure(figsize=(8, 5))
        for i, perm in enumerate(self.permutations):
            plt.scatter(self.times[i], self.scores[i], color=perm_color_map[perm], label=f'{perm[0]} | {perm[1]}')

        plt.title('Scores vs Time by Permutation')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Score')
        plt.legend(title='Permutation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig("scores_vs_time.png")
        plt.show()
