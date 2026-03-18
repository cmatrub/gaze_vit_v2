import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]
wildlife_y_true = np.random.randint(0, 4, size=10)


def plot_confusion_matrix(cm, class_names, step):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    ticks = np.arange(len(class_names))
    ax.set(
        xticks=ticks, yticks=ticks,
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title=f"Confusion Matrix (step {step})",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    w, h = fig.canvas.get_width_height(physical=True)
    return buf.reshape(h, w, 4)[:, :, :3]  # drop alpha


with wandb.init(project="wildlife_classification") as run:
    frames = []
    for step in range(20):
        wildlife_probs = np.random.rand(10, 4)
        wildlife_probs = np.exp(wildlife_probs) / np.sum(
            np.exp(wildlife_probs), axis=1, keepdims=True
        )
        preds = np.argmax(wildlife_probs, axis=1)

        cm = sk_confusion_matrix(wildlife_y_true, preds, labels=range(4))
        fig = plot_confusion_matrix(cm, wildlife_class_names, step)

        frames.append(fig_to_array(fig))
        run.log({"confusion_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)

    video = np.stack(frames).transpose(0, 3, 1, 2)
    run.log({"confusion_matrix_video": wandb.Video(video, fps=2, format="mp4")})
