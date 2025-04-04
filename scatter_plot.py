import csv
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def c_to_p(s):
    # Comma to point
    return s.replace(",", ".")

csv_file = "./dynamic_sparsity.csv"
data = []
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file, ["model_id", "model_size", "weights", "int4_ratio", "group_size", "is_sparse",
                                   "up_ratio", "gate_ratio", "down_ratio", "first_latency", "second_avg_latency", "acc"])
    for row in reader:
        data.append(row)
    data = data[1:]

model_id_to_marker = {
    "Llama-3.1-8B-Instruct": "o",
    "Phi-4-mini-instruct": "s",
    "Qwen2.5-7B-Instruct": "v",
}

point_per_marker = defaultdict(list)
for row in data:
    model_id = row['model_id']
    int4_ratio = float(c_to_p(row["int4_ratio"]))
    group_size = int(c_to_p(row["group_size"]))
    is_sparse = row["is_sparse"] == "True"
    second_avg_latency = float(c_to_p(row["second_avg_latency"]))
    acc = float(c_to_p(row["acc"]))
    point = {
        "x": second_avg_latency,
        "y": acc,
        "c": "r" if is_sparse else "b",
        "s": float(c_to_p(row["model_size"])),
        # "label": model_id,
        "hover_label": f"({int(second_avg_latency)}ms, {acc*100:.2f}%) {model_id} {row['weights']}",
    }
    if row['weights'] == "int4":
        point["hover_label"] += f" r{int4_ratio:.2f}"
        point["hover_label"] += f" gs{group_size}"
    if is_sparse:
        point["hover_label"] += (f"\nup-{c_to_p(row['up_ratio'])[:3]}"
                                 f"_gate-{c_to_p(row['gate_ratio'])[:3]}"
                                 f"_down-{c_to_p(row['down_ratio'])[:3]}")

    point_per_marker[model_id_to_marker[model_id]].append(point)

fig, ax = plt.subplots()
plot_per_marker = {}
for marker, points in point_per_marker.items():
    plot_per_marker[marker] = plt.scatter(
        x=[point["x"] for point in points],
        y=[point["y"] for point in points],
        c=[point["c"] for point in points],
        s=[point["s"] * 10 for point in points],
        marker=marker,
        alpha=0.5,
        # label=points[0]["label"]
    )

annotation_per_marker = {}
for marker in point_per_marker:
    annotation_per_marker[marker] = ax.annotate(
        "",
        fontsize=8,
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annotation_per_marker[marker].set_visible(False)

def update_annotation(ind, marker):
    idx = ind["ind"][0]
    annot = annotation_per_marker[marker]
    pos = plot_per_marker[marker].get_offsets()[idx]
    annot.xy = pos
    text = point_per_marker[marker][idx]["hover_label"]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    for marker in point_per_marker:
        plot = plot_per_marker[marker]
        annot = annotation_per_marker[marker]
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = plot.contains(event)
            if cont:
                update_annotation(ind, marker)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

# Legend
legend_handles = []
for model_id, marker in model_id_to_marker.items():
    for color in ["b", "r"]:
        label = f"{model_id} with sparsity" if color == "r" else f"{model_id} w/o sparsity"
        legend_handles.append(mlines.Line2D([], [], color=color, alpha=0.5, marker=marker, linestyle='None', markersize=10, label=label))
plt.legend(handles=legend_handles)

plt.xlabel("Second token latency")
plt.ylabel("MMLU Accuracy")
plt.grid()
plt.show()
