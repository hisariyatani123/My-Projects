#!/usr/bin/env python3
# Importing all the essential libraries including the messages as well
import rospy
from final_project.msg import ProbabilityData

rospy.init_node('probability_generator', anonymous=True)
publish_time = rospy.Publisher('Time_prob', ProbabilityData, queue_size=10)
publish_session = rospy.Publisher('Session_prob', ProbabilityData, queue_size=10)

def publish_probability_data(publisher, data_dict):
    for key, value in data_dict.items():
        probability_data_msg = ProbabilityData()
        probability_data_msg.time = key
        probability_data_msg.probability = value
        publisher.publish(probability_data_msg)

prob_for_time = {
    3: [0.23, 0.59, 0.45, 0.36, 0.65, 0.79, 0.36, 0.39, 0.62, 0.68, 0.76, 0.56],
    5: [0.27, 0.63, 0.49, 0.40, 0.69, 0.83, 0.40, 0.42, 0.66, 0.72, 0.80, 0.60],
    7: [0.40, 0.76, 0.62, 0.53, 0.82, 0.96, 0.53, 0.55, 0.79, 0.85, 0.93, 0.73],
    9: [0.38, 0.74, 0.60, 0.51, 0.80, 0.94, 0.51, 0.53, 0.77, 0.83, 0.91, 0.71],
    15: [0.39, 0.75, 0.61, 0.52, 0.81, 0.95, 0.52, 0.54, 0.78, 0.84, 0.92, 0.72]
}

prob_for_session = {
    1: [0.23, 0.59, 0.45, 0.36, 0.65, 0.79, 0.36, 0.39, 0.62, 0.68, 0.76, 0.56],
    2: [0.33, 0.69, 0.55, 0.46, 0.75, 0.89, 0.46, 0.49, 0.72, 0.78, 0.86, 0.66],
    3: [0.35, 0.71, 0.57, 0.48, 0.77, 0.91, 0.48, 0.51, 0.74, 0.80, 0.88, 0.68],
    4: [0.38, 0.74, 0.60, 0.51, 0.80, 0.94, 0.51, 0.54, 0.77, 0.83, 0.91, 0.71],
    5: [0.40, 0.76, 0.62, 0.53, 0.82, 0.96, 0.53, 0.56, 0.79, 0.85, 0.93, 0.73]
}

# Publish probabilities for time
publish_probability_data(publish_time, prob_for_time)

# Publish probabilities for session
publish_probability_data(publish_session, prob_for_session)

rospy.spin()





# #!/usr/bin/env python3
# #importing all the essential libraries including the messages as well
# import rospy
# from final_project.msg import ProbabilityData

# rospy.init_node('probability_generator', anonymous=True)
# publish_param1 = rospy.Publisher('Time_prob', ProbabilityData, queue_size=10)


# prob_for_time = {
#     3:[0.23, 0.59, 0.45,0.36,0.65,0.79,0.36,0.39,0.62,0.68,0.76,0.56],
#     5:[0.27, 0.63, 0.49,0.40,0.69,0.83,0.40,0.42,0.66,0.72,0.80,0.60],
#     7:[0.40, 0.76, 0.62,0.53,0.82,0.96,0.53,0.55,0.79,0.85,0.93,0.73],
#     9:[0.38, 0.74, 0.60,0.51,0.80,0.94,0.51,0.53,0.77,0.83,0.91,0.71],
#     15:[0.39, 0.75, 0.61,0.52,0.81,0.95,0.52,0.54,0.78,0.84,0.92,0.72]
# }


# prob_for_session={
#     1:[0.23, 0.59, 0.45,0.36,0.65,0.79,0.36,0.39,0.62,0.68,0.76,0.56],
#     2:[0.33, 0.69, 0.55,0.46,0.75,0.89,0.46,0.49,0.72,0.78,0.86,0.66],
#     3:[0.35, 0.71, 0.57,0.48,0.77,0.91,0.48,0.51,0.74,0.80,0.88,0.68],
#     4:[0.38, 0.74, 0.60,0.51,0.80,0.94,0.51,0.54,0.77,0.83,0.91,0.71],
#     5:[0.40, 0.76, 0.62,0.53,0.82,0.96,0.53,0.56,0.79,0.85,0.93,0.73]
# }

# probability_data_msg = ProbabilityData()

# # Fill in the fields of the message with your probability data
# for key, value in prob_for_time.items():
#     probability_data_msg.time = key
#     probability_data_msg.probability = value
# # publish_param1 = rospy.Publisher('time_prob', String, queue_size=10)
# # publish_param2 = rospy.Publisher('session_prob', String, queue_size=10)

# publish_param1.publish(probability_data_msg)

# # publish_param1.publish(prob_for_session)

# rospy.spin()



