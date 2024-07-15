#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_line_graph(sess, time):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig1, ax1 = plt.subplots(2, 5, figsize=(16, 6))

    color_list = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'lightgray', 'pink']

    # Plot session accuracy
    for i, (sess_id, accuracy) in enumerate(sess.items()):
        ax[0].plot(range(1, len(accuracy) + 1), accuracy, marker='o', label=f"Session {sess_id}", color=color_list[i])
        ax[0].set_xlabel('Objects')
        ax[0].set_ylabel('Accuracy (%)')
        ax[0].set_title('Overall Accuracies for Different Objects (Session)')
        ax[0].grid(True)
        ax[0].legend()

        objects = range(1, len(accuracy) + 1)
        ax1[0, i].bar(objects, accuracy, color=color_list[i])
        ax1[0, i].set_title(f'Object Accuracies for Session {sess_id}')
        ax1[0, i].set_xlabel('Object')
        ax1[0, i].set_ylabel('Accuracy (%)')
        ax1[0, i].set_xticks(objects)

    # Plot time accuracy
    for i, (time_id, accuracy) in enumerate(time.items()):
        ax[1].plot(range(3, 16, 2), accuracy, marker='o', label=f"Time {time_id}", color=color_list[i])
        ax[1].set_xlabel('Time Intervals')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_title('Overall Accuracies for Different Objects (Time)')
        ax[1].set_xticks(range(3, 16, 2))
        ax[1].grid(True)
        ax[1].legend()

        objects = range(1, len(accuracy) + 1)
        ax1[1, i].bar(objects, accuracy, color=color_list[i])
        ax1[1, i].set_title(f'Object Accuracies for Time {time_id}')
        ax1[1, i].set_xlabel('Object')
        ax1[1, i].set_ylabel('Accuracy (%)')
        ax1[1, i].set_xticks(objects)

    # Save the plots as images
    fig.tight_layout()
    fig.savefig('/home/hisariya/catkin_ws/src/final_project/scripts/session_accuracy_plot.png')
    fig1.tight_layout()
    fig1.savefig('/home/hisariya/catkin_ws/src/final_project/scripts/time_accuracy_plot.png')

def session_accuracy_callback(data):
    sess_acc_data = data.data
    session_acc = {}
    for i, acc in enumerate(sess_acc_data):
        session_acc[i + 1] = acc
    return session_acc

def time_accuracy_callback(data):
    time_acc_data = data.data
    time_acc = {}
    for i, acc in enumerate(time_acc_data):
        time_acc[(i + 1) * 2 + 1] = acc
    return time_acc

def main():
    rospy.init_node('accuracy_plotter')

    # Define dictionaries to store accuracy data for session and time
    session_accuracy = {}
    time_accuracy = {}

    # Define callback functions to parse accuracy data from messages and update dictionaries
    def session_accuracy_callback(data):
        nonlocal session_accuracy
        session_accuracy = session_accuracy_callback(data)

    def time_accuracy_callback(data):
        nonlocal time_accuracy
        time_accuracy = time_accuracy_callback(data)

    # Subscribe to topics for session and time accuracy
    rospy.Subscriber('session_accuracy_topic', Float32MultiArray, session_accuracy_callback)
    rospy.Subscriber('time_accuracy_topic', Float32MultiArray, time_accuracy_callback)

    # Spin to keep the node running
    rospy.spin()

    # After spin, when the node is shutting down, plot the accuracy data
    plot_line_graph(session_accuracy, time_accuracy)

if __name__ == '__main__':
    main()





# #!/usr/bin/env python3
# import rospy
# from std_msgs.msg import Float32MultiArray
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

# def plot_line_graph(sess, time):
#     _, ax = plt.subplots(1, 2, figsize=(16, 6))
#     _, ax1 = plt.subplots(2, 5, figsize=(16, 6))

#     color_list = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'lightgray', 'pink']

#     # Plot session accuracy
#     for i, (sess_id, accuracy) in enumerate(sess.items()):
#         ax[0].plot(range(1, len(accuracy) + 1), accuracy, marker='o', label=f"Session {sess_id}", color=color_list[i])
#         ax[0].set_xlabel('Objects')
#         ax[0].set_ylabel('Accuracy (%)')
#         ax[0].set_title('Overall Accuracies for Different Objects (Session)')
#         ax[0].grid(True)
#         ax[0].legend()

#         objects = range(1, len(accuracy) + 1)
#         ax1[0, i].bar(objects, accuracy, color=color_list[i])
#         ax1[0, i].set_title(f'Object Accuracies for Session {sess_id}')
#         ax1[0, i].set_xlabel('Object')
#         ax1[0, i].set_ylabel('Accuracy (%)')
#         ax1[0, i].set_xticks(objects)

#     # Plot time accuracy
#     for i, (time_id, accuracy) in enumerate(time.items()):
#         ax[1].plot(range(3, 16, 2), accuracy, marker='o', label=f"Time {time_id}", color=color_list[i])
#         ax[1].set_xlabel('Time Intervals')
#         ax[1].set_ylabel('Accuracy (%)')
#         ax[1].set_title('Overall Accuracies for Different Objects (Time)')
#         ax[1].set_xticks(range(3, 16, 2))
#         ax[1].grid(True)
#         ax[1].legend()

#         objects = range(1, len(accuracy) + 1)
#         ax1[1, i].bar(objects, accuracy, color=color_list[i])
#         ax1[1, i].set_title(f'Object Accuracies for Time {time_id}')
#         ax1[1, i].set_xlabel('Object')
#         ax1[1, i].set_ylabel('Accuracy (%)')
#         ax1[1, i].set_xticks(objects)

#     # Display the plot
#     plt.tight_layout()
#     plt.show()

# def session_accuracy_callback(data):
#     sess_acc_data = data.data
#     session_acc = {}
#     for i, acc in enumerate(sess_acc_data):
#         session_acc[i + 1] = acc
#     return session_acc

# def time_accuracy_callback(data):
#     time_acc_data = data.data
#     time_acc = {}
#     for i, acc in enumerate(time_acc_data):
#         time_acc[(i + 1) * 2 + 1] = acc
#     return time_acc

# def main():
#     rospy.init_node('accuracy_plotter')

#     # Define dictionaries to store accuracy data for session and time
#     session_accuracy = {}
#     time_accuracy = {}

#     # Define callback functions to parse accuracy data from messages and update dictionaries
#     def session_accuracy_callback(data):
#         nonlocal session_accuracy
#         session_accuracy = session_accuracy_callback(data)

#     def time_accuracy_callback(data):
#         nonlocal time_accuracy
#         time_accuracy = time_accuracy_callback(data)

#     # Subscribe to topics for session and time accuracy
#     rospy.Subscriber('session_accuracy_topic', Float32MultiArray, session_accuracy_callback)
#     rospy.Subscriber('time_accuracy_topic', Float32MultiArray, time_accuracy_callback)

#     # Spin to keep the node running
#     rospy.spin()

#     # After spin, when the node is shutting down, plot the accuracy data
#     plot_line_graph(session_accuracy, time_accuracy)

# if __name__ == '__main__':
#     main()









# # #!/usr/bin/env python3
# # #----importing rospy and essential files----
# # import rospy
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def plot_line_graph(acc):
# #     # data_transposed = np.array(acc).T.tolist()

# #     # Plotting
# #     plt.figure(figsize=(10, 6))

# #     # for i, session in enumerate(data_transposed):
# #     #     plt.plot(range(1, 6), session, marker='o', label=f"Element {i+1}")

# #     k = list(acc.keys())
# #     v = list(acc.values())

# #     print(k, v)
# #     for i, _ in enumerate(acc):
# #         # print(i)
# #         plt.plot(k, v, marker='o', label=f"Element {k[i]}")

# #     plt.xlabel('Sessions')
# #     plt.ylabel('Overall Accuracy')
# #     plt.title('Overall Accuracies for Different Objects')
# #     plt.xticks(range(1, 6))
# #     plt.legend()
# #     plt.grid(True)
# #     plt.tight_layout()

# #     # Display the plot
# #     plt.show()