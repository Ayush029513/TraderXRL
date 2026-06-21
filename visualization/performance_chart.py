import matplotlib.pyplot as plt

strategies = [
    "Buy & Hold",
    "PPO Agent"
]

returns = [
    325.57,
    199.20
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    strategies,
    returns
)

plt.title(
    "Strategy Performance Comparison"
)

plt.ylabel(
    "Return (%)"
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{bar.get_height():.1f}%",
        ha="center"
    )

plt.grid(axis="y")

plt.savefig(
    "results/performance_comparison.png"
)

plt.show()