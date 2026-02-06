import matplotlib
matplotlib.use('Agg') # Prevents macOS GUI errors
import matplotlib.pyplot as plt

# Data from your Feature 111 inspection
data = {
    "' directed'": 41.55,
    "' thought'": 34.50,
    "' one'": 30.69,
    "' him'": 30.26,
    "' emergence'": 23.25,
    "' surprise'": 12.99,
    "' belong'": 12.93,
    "' Robert'": 9.88,
    "' however'": 4.92
}

# Sorting data for the graph
tokens = list(data.keys())[::-1]
values = list(data.values())[::-1]

plt.figure(figsize=(10, 6))
# Professional color scheme: High activation is dark, lower is light
colors = ['skyblue' if v < 30 else 'royalblue' for v in values]
bars = plt.barh(tokens, values, color=colors)

plt.xlabel('Activation Strength', fontsize=12)
plt.title('Archetypal SAE: Semantic Profile of Feature 111 (Layer 9)', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.1f}', va='center')

plt.tight_layout()
plt.savefig('feature_111_profile.png')
print("--- SUCCESS: Graph saved as 'feature_111_profile.png' ---")