#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from prob import prob_for_time, prob_for_session
from run_experiments import calc_acc
from plots import plot_line_graph

def main():
    rospy.init_node('main_node', anonymous=True)
    session_pub = rospy.Publisher('session_accuracy', String, queue_size=10)
    time_pub = rospy.Publisher('time_accuracy', String, queue_size=10)

    session_acc = {}
    time_acc = {}

    for k, v in prob_for_session.items():
        session_acc[k] = calc_acc(v, k, exp_type="session")
        session_pub.publish(str(session_acc))

    for k, v in prob_for_time.items():
        time_acc[k] = calc_acc(v, k, exp_type="time interval")
        time_pub.publish(str(time_acc))

    print(f"\nSession Output: \n{session_acc}")
    print(f"\nTime Output: \n{time_acc}")

    rospy.spin()

if __name__ == '__main__':
    main()





# #!/usr/bin/env python3
# import rospy

# from prob import prob_for_time, prob_for_session
# from run_experiments import calc_acc
# from plots import plot_line_graph

# rospy.init_node('main_node', anonymous=True)
# # session_pub = rospy.Publisher('session_accuracy', String, queue_size=10)
# # time_pub = rospy.Publisher('time_accuracy', String, queue_size=10)

# session_acc = {}
# time_acc = {}

# for k, v in prob_for_session.items():
#     session_acc[k] = calc_acc(v, k, exp_type="session")
#     # session_pub.publish(str(session_acc))

# for k, v in prob_for_time.items():
#     time_acc[k] = calc_acc(v, k, exp_type="time intervel")
#     # time_pub.publish(str(time_acc))
# print(f"\nSession Ouput: \n{session_acc}")

# print(f"\nTime Ouput: \n{time_acc}")



# # plotting results 
# plot_line_graph(acc=session_acc)
# rospy.spin()

