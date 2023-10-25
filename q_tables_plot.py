from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import cv2
import os
from q_learning import Action
from main import NUM_EPISODES, SHOW_EVERY

style.use('ggplot')

if not os.path.exists("qtable_charts"):
    os.makedirs("qtable_charts")


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


fig = plt.figure(figsize=(12, 12))


for i in range(0, NUM_EPISODES, SHOW_EVERY):
    print(i)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    q_table = np.load(f"qtables/qtable_{i}.npy")

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(y, x, c=get_q_color(y_vals[Action.UP.value], y_vals)[
                        0], marker="o", alpha=get_q_color(y_vals[Action.UP.value], y_vals)[1])
            ax2.scatter(y, x, c=get_q_color(y_vals[Action.DOWN.value], y_vals)[
                        0], marker="o", alpha=get_q_color(y_vals[Action.DOWN.value], y_vals)[1])
            ax3.scatter(y, x, c=get_q_color(y_vals[Action.LEFT.value], y_vals)[
                        0], marker="o", alpha=get_q_color(y_vals[Action.LEFT.value], y_vals)[1])
            ax4.scatter(y, x, c=get_q_color(y_vals[Action.RIGHT.value], y_vals)[
                        0], marker="o", alpha=get_q_color(y_vals[Action.RIGHT.value], y_vals)[1])

        # invert y_Axis
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax1.set_ylabel("Action UP")
    ax2.set_ylabel("Action DOWN")
    ax3.set_ylabel("Action LEFT")
    ax4.set_ylabel("Action RIGHT")

    # plt.show()
    plt.savefig(f"qtable_charts/chart_ep_{i}.png")
    plt.clf()


def make_video():
    # windows:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Linux:
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('qtable_charts/qlearn.avi',
                          fourcc, 1.0, (1200, 1200)) # Change to 60fps if the SHOW_EVERY is shorter

    for i in range(0, NUM_EPISODES, SHOW_EVERY):
        img_path = f"qtable_charts/chart_ep_{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video()
