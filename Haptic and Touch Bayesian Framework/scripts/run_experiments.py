#!/usr/bin/env python3
import rospy
import random
from final_project.msg import ProbabilityData
from std_msgs.msg import Float32MultiArray

def calc_acc(probability, sess_id, exp_type):
    acc = []

    print(f"\nFOR {exp_type} {sess_id}:")

    for i, chosen_probability in enumerate(probability):
        # Total number of attempts
        experiments = 2500

        # Simulate the process
        true_positive = 0
        for _ in range(experiments):
            # Simulate an attempt for the chosen element
            if random.random() < chosen_probability:
                true_positive += 1

        # Calculate overall accuracy for the chosen element
        accuracy = true_positive / experiments
        acc.append(round(accuracy*100, 2))
        print(f"Overall accuracy for Object {i+1} over {experiments} attempts: {accuracy * 100:.2f}%")

    return acc

def probability_callback(data, exp_type):
    # Extract data from the received message
    probability = data.probability
    sess_id = data.time

    # Calculate accuracy
    accuracy = calc_acc(probability, sess_id, exp_type)

    # Create a message to publish accuracy
    accuracy_msg = Float32MultiArray()
    accuracy_msg.data = accuracy

    # Publish the accuracy data
    if exp_type == "Session":
        session_accuracy_publisher.publish(accuracy_msg)
    elif exp_type == "Time":
        time_accuracy_publisher.publish(accuracy_msg)

def main():
    # Initialize the ROS node
    rospy.init_node('accuracy_calculator')

    # Subscribe to the session probability topic
    rospy.Subscriber('Session_prob', ProbabilityData, probability_callback, callback_args="Session")

    # Subscribe to the time probability topic
    rospy.Subscriber('Time_prob', ProbabilityData, probability_callback, callback_args="Time")

    # Create publishers for accuracy data
    global session_accuracy_publisher, time_accuracy_publisher
    session_accuracy_publisher = rospy.Publisher('session_accuracy_topic', Float32MultiArray, queue_size=10)
    time_accuracy_publisher = rospy.Publisher('time_accuracy_topic', Float32MultiArray, queue_size=10)

    # Spin to keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# import rospy
# import random
# from final_project.msg import ProbabilityData
# from std_msgs.msg import Float32MultiArray

# def calc_acc(probability, sess_id, exp_type):
#     acc = []

#     print(f"\nFOR {exp_type} {sess_id}:")

#     for i, chosen_probability in enumerate(probability):
#         # Total number of attempts
#         experiments = 2500

#         # Simulate the process
#         true_positive = 0
#         for _ in range(experiments):
#             # Simulate an attempt for the chosen element
#             if random.random() < chosen_probability:
#                 true_positive += 1

#         # Calculate overall accuracy for the chosen element
#         accuracy = true_positive / experiments
#         acc.append(round(accuracy*100, 2))
#         print(f"Overall accuracy for Object {i+1} over {experiments} attempts: {accuracy * 100:.2f}%")

#     return acc

# def probability_callback(data):
#     # Extract data from the received message
#     probability = data.probability
#     sess_id = data.time

#     # Calculate accuracy for session probability
#     session_accuracy = calc_acc(probability, sess_id, "Session")

#     # Create a message to publish accuracy
#     accuracy_msg = Float32MultiArray()
#     accuracy_msg.data = session_accuracy

#     # Publish the accuracy data
#     accuracy_publisher.publish(accuracy_msg)

# def main():
#     # Initialize the ROS node
#     rospy.init_node('accuracy_calculator')

#     # Subscribe to the session probability topic
#     rospy.Subscriber('Session_prob', ProbabilityData, probability_callback)

#     # Create a publisher for accuracy data
#     global accuracy_publisher
#     accuracy_publisher = rospy.Publisher('accuracy_topic', Float32MultiArray, queue_size=10)

#     # Spin to keep the node running
#     rospy.spin()

# if __name__ == '__main__':
#     main()





# #!/usr/bin/env python3
# import random

# def calc_acc(probability, sess_id, exp_type):
#     acc=[]

#     print(f"\nFOR {exp_type} {sess_id}:")
    
#     for i in range(12):
#         # chosen_probability = prob_1[element_number - 1]
#         chosen_probability = probability[i]

#         # Total number of attempts
#         experiments = 2500

#         # Simulate the process
#         true_positive = 0
#         for _ in range(experiments):
#         # Simulate an attempt for the chosen element
#             if random.random() < chosen_probability:
#                 true_positive += 1

#         # Calculate overall accuracy for the chosen element
#         accuracy = true_positive / experiments
#         acc.append(round(accuracy*100,2))
#         print(f"Overall accuracy for Object {i+1} over {experiments} attempts: {accuracy * 100:.2f}%")

#     return acc
