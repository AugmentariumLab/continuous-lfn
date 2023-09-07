import os
import json
import sys

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_benchmark_results(csv_path):
  with open(csv_path, "r") as f:
    return json.load(f)

resolution = "1008_760"

run_dir = sys.argv[1]
print("Run dir: ", run_dir)
slimnet_results = load_benchmark_results(os.path.join(run_dir, "benchmark_slimnet.json"))
num_lods = 462 if "461" in slimnet_results[resolution] else 385
slimnet_rendertimes = [1000 * slimnet_results[resolution][str(i)] for i in range(num_lods)]
slimnet_widths = [((512 - num_lods + 1) + i) for i in range(num_lods)]

widths = slimnet_widths
rendertimes = slimnet_rendertimes
parity = np.array(["Odd" for _ in range(len(slimnet_widths))], dtype="<U12")
parity[::2] = "Even"
parity[::8] = "Div by 8"

df = pd.DataFrame({
    "widths": widths,
    "rendertimes": rendertimes,
    "parity": parity
})

f, ax = plt.subplots(figsize=(5, 3))
sns.set_theme(style="ticks")
ax = sns.scatterplot(x="widths", y="rendertimes",
                hue="parity", 
                palette="tab10",
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)
ax.set_xticks((128 * np.arange(1,5)))
plt.title("Render Time Across Levels of Detail")
plt.xlabel("Model Width")
plt.ylabel("Render Time (ms)")
# Remove title from legend
ax.legend(title="")
output_file=os.path.join(run_dir, "rendertimeparity_plot.png")
plt.savefig(output_file, bbox_inches='tight', dpi=300)