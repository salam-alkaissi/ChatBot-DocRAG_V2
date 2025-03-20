import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import io

def create_matplotlib_plot(title):
    # Create matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 2], 'o-')
    ax.set_title(title)
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Important: clean up memory
    return img

def generate_keyword_table(keywords: list, counts: list) -> str:
    """Generate Markdown-formatted keyword frequency table."""
    if not keywords or not counts:
        return "No keywords found"
    
    table = "| Keyword | Count |\n|---------|-------|\n"
    for kw, cnt in zip(keywords, counts):
        table += f"| {kw} | {cnt} |\n"
    return table

def generate_bar_chart(keywords: list, counts: list) -> str:
    """Generate horizontal bar chart of keyword frequencies."""
    if not keywords or not counts:
        return None
    
    # Create outputs directory if not exists
    os.makedirs("outputs", exist_ok=True)
    
    # Sort by frequency descending
    sorted_indices = np.argsort(counts)[::-1]
    keywords = [keywords[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Dynamic figure sizing
    fig_height = max(4, len(keywords) * 0.5)
    plt.figure(figsize=(10, fig_height))
    
    y_pos = np.arange(len(keywords))
    plt.barh(y_pos, counts, align='center', color='skyblue')
    plt.yticks(y_pos, labels=keywords)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top Keyword Frequencies', fontsize=14, pad=20)
    
    plt.gca().invert_yaxis()  # Highest frequency at top
    plt.tight_layout()
    
    chart_path = "outputs/keywords.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()  # Prevent memory leaks
    
    return chart_path