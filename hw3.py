#!/usr/bin/env python

from copy import deepcopy
import math
import numpy
import random
from threading import Thread, Lock
import sys
import matplotlib.pyplot as plt

import actionlib
import control_msgs.msg
import geometry_msgs.msg
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import sensor_msgs.msg
import tf
import trajectory_msgs.msg
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker

def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

def convert_from_message(msg):
    R = tf.transformations.quaternion_matrix((msg.orientation.x,
                                              msg.orientation.y,
                                              msg.orientation.z,
                                              msg.orientation.w))
    T = tf.transformations.translation_matrix((msg.position.x, 
                                               msg.position.y, 
                                               msg.position.z))
    return numpy.dot(T,R)

class RRTNode(object):
    def __init__(self):
        self.q=numpy.zeros(7)
        self.parent = None

class MoveArm(object):

    def __init__(self):
        print "HW3 initializing..."
        # Prepare the mutex for synchronization
        self.mutex = Lock()

        # min and max joint values are not read in Python urdf, so we must hard-code them here
        self.q_min = []
        self.q_max = []
        self.q_min.append(-1.700);self.q_max.append(1.700)
        self.q_min.append(-2.147);self.q_max.append(1.047)
        self.q_min.append(-3.054);self.q_max.append(3.054)
        self.q_min.append(-0.050);self.q_max.append(2.618)
        self.q_min.append(-3.059);self.q_max.append(3.059)
        self.q_min.append(-1.570);self.q_max.append(2.094)
        self.q_min.append(-3.059);self.q_max.append(3.059)

        # Subscribes to information about what the current joint values are.
        rospy.Subscriber("robot/joint_states", sensor_msgs.msg.JointState, self.joint_states_callback)

        # Initialize variables
        self.q_current = []
        self.joint_state = sensor_msgs.msg.JointState()

        # Create interactive marker
        self.init_marker()

        # Connect to trajectory execution action
        self.trajectory_client = actionlib.SimpleActionClient('/robot/limb/left/follow_joint_trajectory', 
                                                              control_msgs.msg.FollowJointTrajectoryAction)
        self.trajectory_client.wait_for_server()
        print "Joint trajectory client connected"

        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print "IK service ready"

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print "State validity service ready"

        # Initialize MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm") 
        print "MoveIt! interface ready"

        # How finely to sample each joint
        self.q_sample = [0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.4]
        self.joint_names = ["left_s0", "left_s1",
                            "left_e0", "left_e1",
                            "left_w0", "left_w1","left_w2"]

        # Options
        self.subsample_trajectory = True
        self.spline_timing = True
        self.show_plots = False

        print "Initialization done."


    def control_marker_feedback(self, feedback):
        pass

    def get_joint_val(self, joint_state, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
            return 0
        i = joint_state.name.index(name)
        return joint_state.position[i]

    def set_joint_val(self, joint_state, q, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
        i = joint_state.name.index(name)
        joint_state.position[i] = q

    """ Given a complete joint_state data structure, this function finds the values for 
    a particular set of joints in a particular order (in our case, the left arm joints ordered
    from proximal to distal) and returns a list q[] containing just those values.
    """
    def q_from_joint_state(self, joint_state):
        q = []
        q.append(self.get_joint_val(joint_state, "left_s0"))
        q.append(self.get_joint_val(joint_state, "left_s1"))
        q.append(self.get_joint_val(joint_state, "left_e0"))
        q.append(self.get_joint_val(joint_state, "left_e1"))
        q.append(self.get_joint_val(joint_state, "left_w0"))
        q.append(self.get_joint_val(joint_state, "left_w1"))
        q.append(self.get_joint_val(joint_state, "left_w2"))
        return q

    """ Given a list q[] of joint values and an already populated joint_state, this function assumes 
    that the passed in values are for a particular set of joints in a particular order (in our case,
    the left arm joints ordered from proximal to distal) and edits the joint_state data structure to
    set the values to the ones passed in.
    """
    def joint_state_from_q(self, joint_state, q):
        self.set_joint_val(joint_state, q[0], "left_s0")
        self.set_joint_val(joint_state, q[1], "left_s1")
        self.set_joint_val(joint_state, q[2], "left_e0")
        self.set_joint_val(joint_state, q[3], "left_e1")
        self.set_joint_val(joint_state, q[4], "left_w0")
        self.set_joint_val(joint_state, q[5], "left_w1")
        self.set_joint_val(joint_state, q[6], "left_w2")        

    """ Creates simple timing information for a trajectory, where each point has velocity
    and acceleration 0 for all joints, and all segments take the same amount of time
    to execute.
    """
    def compute_simple_timing(self, q_list, time_per_segment):
        v_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        a_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        t = [i*time_per_segment for i in range(0,len(q_list))]
        return v_list, a_list, t

    """ This function will perform IK for a given transform T of the end-effector. It returs a list q[]
    of 7 values, which are the result positions for the 7 joints of the left arm, ordered from proximal
    to distal. If no IK solution is found, it returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = "left_arm"
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = "base"
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = self.q_from_joint_state(res.solution.joint_state)
        return q

    """ This function checks if a set of joint angles q[] creates a valid state, or one that is free
    of collisions. The values in q[] are assumed to be values for the joints of the left arm, ordered
    from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = "left_arm"
        current_joint_state = deepcopy(self.joint_state)
        current_joint_state.position = list(current_joint_state.position)
        self.joint_state_from_q(current_joint_state, q)
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state = current_joint_state
        res = self.state_valid_service(req)
        return res.valid

    """ this function checks points on segment between two nodes with a precision of 0.1
    """
    def is_segment_valid(self,start_node,end_node):
	test_segment = end_node-start_node
	test_segment_norm = numpy.linalg.norm(test_segment)
	if test_segment_norm < 0.1:
		if self.is_state_valid(start_node) and self.is_state_valid(end_node):
			return True
		else:
			return False
	n = int(round(test_segment_norm/0.1)) + 2
	gap = test_segment/n
	for i in range(n+1):
		test_node = start_node + i*gap
		if not self.is_state_valid(test_node):
			return False
	return True

    def compute_simple_timing_own(self,resample_path,time_per_segment):#discarded
	i=0
	v0=0
	a0=0
	t0=0
	v1=0
	a1=0
	t1=0
	t_j = len(resample_path)-1
	a=4*t_j
	coeffs = numpy.zeros((7,a))
	v_list = numpy.zeros((len(resample_path),7))
	a_list = numpy.zeros((len(resample_path),7))
	t_list = []
	

	for k in range(7):
		while(i!=t_j-1):
			
			coeffs[k][4*i+3]=resample_path[i][k]
			coeffs[k][4*i+2]=v0
			coeffs[k][4*i+1]=a0*0.5
			coeffs[k][4*i]=resample_path[i+1][k]-resample_path[i][k]-v0-a0*0.5
			
			v_list[i][k]=v0
			a_list[i][k]=a0

			v1=3*coeffs[k][4*i]+2*coeffs[k][4*i+1]+coeffs[k][4*i+2]
			a1=6*coeffs[k][4*i]+2*coeffs[k][4*i+1]

			v0=v1
			a0=a1
			i=i+1
                
                #a_list[i][k]=a0
		#v_list[i][k]=v0
		coeffs[k][4*i+3]=resample_path[i][k]
		coeffs[k][4*i+2]=v0
		coeffs[k][4*i+1]=3*resample_path[i+1][k]-2*v0-3*resample_path[i][k]
		coeffs[k][4*i]=-2*resample_path[i+1][k]+v0+2*resample_path[i][k]		
		
		a_list[i+1][k]=6*coeffs[k][4*i]+2*coeffs[k][4*i+1]
		v_list[i+1][k]=0	
	
		#m = numpy.zeros(4,4)
		#m[0,3]=m[1,3]=m[2,2]=m[3,2]=1
		#for i in range(3):
		#	m[1,2-i]=T**(i+1)
		#	m[3,2-i]=(i+1)*(T**i)

		#m_inv = numpy.linalg.inv(m)
		#temp = numpy.dot(m_inv,q_l_seg)
		#coesffs_l.append(temp)
		#j = j+1

      	#t_list = [i*time_per_segment for i in range(0,len(resample_path))]
	for i in range(0,len(resample_path)):
		t_list.append(1)
	return coeffs,v_list,a_list,t_list

    def compute_simple_timing_own1(self,resample_path,time_per_segment):#discarded
        
        t=[]
        coeffs=[]
        #for i in range(0,len(resample_path)):
	#	t.append(1)
        t = [i*time_per_segment for i in range(0,len(resample_path))]
        a=len(resample_path)-1

        v_list=numpy.zeros((a+1,7))
        a_list=numpy.zeros((a+1,7))
        m=numpy.zeros((4*a,4*a))
        
	m_inv=[]
        for h in range(7):
        	list1=[]

		i=0
        	j=0
        	k=0
        	while(i!=2*a):
            
                        m[i][j]=t[k]**3
                        m[i][j+1]=t[k]**2
                        m[i][j+2]=t[k]
                        m[i][j+3]=1
			print i,j,k
                        m[i+1][j]=t[k+1]**3
                        m[i+1][j+1]=t[k+1]**2
                        m[i+1][j+2]=t[k+1]
                        m[i+1][j+3]=1

                        list1.append(resample_path[k][h])
                        list1.append(resample_path[k+1][h])

                        i=i+2
                        j=j+4
                        k=k+1

		print list1                
		j=0
                k=1
                while(i!=3*a-1):

                	m[i][j]=3*(t[k]**2)
                        m[i][j+1]=t[k]
                        m[i][j+2]=1

                        m[i][j+4]=-3*(t[k]**2)
                        m[i][j+5]=-t[k]
                        m[i][j+6]=-1

                        list1.append(0)

                        i=i+1
                        j=j+4
                        k=k+1

                j=0
                k=1
                while(i!=4*a-2):

                	m[i][j]=6*t[k]
                	m[i][j+1]=2

                        m[i][j+4]=-6*t[k]
                        m[i][j+5]=-2

			list1.append(0)

                        i=i+1
			j=j+4
			k=k+1

		k=len(t)-1
		m[i][2]=1
		m[i+1][4*a-4]=3*(t[k]**2)
		m[i+1][4*a-3]=2*t[k]
		m[i+1][4*a-2]=1
		print m
		print len(m)
                list1.append(0)
                list1.append(0)
		print list1
		m_inv=numpy.linalg.inv(m)
		coeffs0=numpy.dot(m_inv,list1)
		coeffs.append(coeffs0)
	print coeffs
	print len(coeffs)
	print len(coeffs0)
        
        
        for j in range(7):
		k=1
		p=4
                for i in range(1,len(resample_path)-1):

                        v_list[i][j]=3*coeffs[j][p]*(t[k]**2)+2*coeffs[j][p+1]*t[k]+coeffs[j][p+2]
			k=k+1
			p=p+4
       
        for j in range(7):
		k=1
		p=4
                for i in range(1,len(resample_path)-1):

                        a_list[i][j]=6*coeffs[j][p]*t[k]+2*coeffs[j][p+1]
			k=k+1
			p=p+4
		
		a_list[0][j]=2*coeffs[j][1]
		a_list[a][j]=6*coeffs[j][4*a-4]*t[a]+2*coeffs[j][4*a-3]
			
	return coeffs,v_list,a_list,t

    def compute_simple_timing_own2(self,resample_path,time_per_segment):
        
        t=[]
        coeffs=[]
        #for i in range(0,len(resample_path)):
	#	t.append(1)
        t = [i*time_per_segment for i in range(0,len(resample_path))]
        a=len(resample_path)-1

        v_list=numpy.zeros((a+1,7))
        a_list=numpy.zeros((a+1,7))
        m=numpy.zeros((4*a,4*a))
        
	m_inv=[]
        for h in range(7):
        	list1=[]

		i=0
        	j=0
        	k=0
        	while(i!=2*a):
            
                        m[i][j]=0
                        m[i][j+1]=0
                        m[i][j+2]=0
                        m[i][j+3]=1

                        m[i+1][j]=1
                        m[i+1][j+1]=1
                        m[i+1][j+2]=1
                        m[i+1][j+3]=1

                        list1.append(resample_path[k][h])
                        list1.append(resample_path[k+1][h])

                        i=i+2
                        j=j+4
                        k=k+1

		#print list1                
		j=0
                k=1
                while(i!=3*a-1):

                	m[i][j]=3
                        m[i][j+1]=2
                        m[i][j+2]=1

                        m[i][j+4]=0
                        m[i][j+5]=0
                        m[i][j+6]=-1

                        list1.append(0)

                        i=i+1
                        j=j+4
                        k=k+1

                j=0
                k=1
                while(i!=4*a-2):

                	m[i][j]=6
                	m[i][j+1]=2

                        m[i][j+4]=0
                        m[i][j+5]=-2

			list1.append(0)

                        i=i+1
			j=j+4
			k=k+1

		k=len(t)-1
		m[i][2]=1
		m[i+1][4*a-4]=3
		m[i+1][4*a-3]=2
		m[i+1][4*a-2]=1
		#print m
		#print len(m)
                list1.append(0)
                list1.append(0)
		#print list1
		m_inv=numpy.linalg.inv(m)
		coeffs0=numpy.dot(m_inv,list1)
		coeffs.append(coeffs0)
	#print coeffs
	print len(coeffs)
	print len(coeffs0)
        
        
        for j in range(7):
		k=1
		p=4
                for i in range(1,len(resample_path)-1):

                        v_list[i][j]=3*coeffs[j][p]+2*coeffs[j][p+1]+coeffs[j][p+2]
			k=k+1
			p=p+4
       
        for j in range(7):
		k=1
		p=4
                for i in range(1,len(resample_path)-1):

                        a_list[i][j]=6*coeffs[j][p]+2*coeffs[j][p+1]
			k=k+1
			p=p+4
		
		a_list[0][j]=2*coeffs[j][1]
		a_list[a][j]=6*coeffs[j][4*a-4]+2*coeffs[j][4*a-3]
			
	return coeffs,v_list,a_list,t
            
            

	
	
   

    # This function will plot the position, velocity and acceleration of a joint
    # based on the polynomial coefficients of each segment that makes up the 
    # trajectory.
    # Arguments:
    # - num_segments: the number of segments in the trajectory
    # - coefficients: the coefficients of a cubic polynomial for each segment, arranged
    #   as follows [a_1, b_1, c_1, d_1, ..., a_n, b_n, c_n, d_n], where n is the number
    #   of segments
    # - time_per_segment: the time (in seconds) allocated to each segment.
    # This function will display three plots. Execution will continue only after all 
    # plot windows have been closed.
    def plot_trajectory(self, num_segments, coeffs, time_per_segment):
        resolution = 1.0e-2
        assert(num_segments*4 == len(coeffs))
        t_vec = []
        q_vec = []
        a_vec = []
        v_vec = []
        #t=0
	#j=1
        for i in range(0,num_segments):
	#i=0
	#while i<num_segments:
	    t=0
            while t<time_per_segment:
                q,a,v = self.sample_polynomial(coeffs,i,t)
                t_vec.append(t+i*time_per_segment)
                q_vec.append(q)
                a_vec.append(a)
                v_vec.append(v)
                t = t+resolution
	    #t=t+time_per_segment
	    #j=j+1
	#	i=i+2
        self.plot_series(t_vec,q_vec,"Position")
        self.plot_series(t_vec,v_vec,"Velocity")
        self.plot_series(t_vec,a_vec,"Acceleration")
        plt.show()


    """ This is the main function to be filled in for HW3.
    Parameters:
    - q_start: the start configuration for the arm
    - q_goal: the goal configuration for the arm
    - q_min and q_max: the min and max values for all the joints in the arm.
    All the above parameters are arrays. Each will have 7 elements, one for each joint in the arm.
    These values correspond to the joints of the arm ordered from proximal (closer to the body) to 
    distal (further from the body). 

    The function must return a trajectory as a tuple (q_list,v_list,a_list,t).
    If the trajectory has n points, then q_list, v_list and a_list must all have n entries. Each
    entry must be an array of size 7, specifying the position, velocity and acceleration for each joint.

    For example, the i-th point of the trajectory is defined by:
    - q_list[i]: an array of 7 numbers specifying position for all joints at trajectory point i
    - v_list[i]: an array of 7 numbers specifying velocity for all joints at trajectory point i
    - a_list[i]: an array of 7 numbers specifying acceleration for all joints at trajectory point i
    Note that q_list, v_list and a_list are all lists of arrays. 
    For example, q_list[i][j] will be the position of the j-th joint (0<j<7) at trajectory point i 
    (0 < i < n).

    For example, a trajectory with just 2 points, starting from all joints at position 0 and 
    ending with all joints at position 1, might look like this:

    q_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([1, 1, 1, 1, 1, 1, 1]) ]
    v_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
    a_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
             
    Note that the trajectory should always begin from the current configuration of the robot.
    Hence, the first entry in q_list should always be equal to q_start. 

    In addition, t must be a list with n entries (where n is the number of points in the trajectory).
    For the i-th trajectory point, t[i] must specify when this point should be reached, relative to
    the start of the trajectory. As a result t[0] should always be 0. For the previous example, if we
    want the second point to be reached 10 seconds after starting the trajectory, we can use:

    t=[0,10]

    When you are done computing all of these, return them using

    return q_list,v_list,a_list,t

    In addition, you can use the function self.is_state_valid(q_test) to test if the joint positions 
    in a given array q_test create a valid (collision-free) state. q_test will be expected to 
    contain 7 elements, each representing a joint position, in the same order as q_start and q_goal.
    """               
    def motion_plan(self, q_start, q_goal, q_min, q_max):
        # ---------------- replace this with your code ------------------
	use_RRT = 1	#Mode change:1 for RRT, 0 for PRM
	#-----------------------RRT--------------------------------------
	new_path = []
	if use_RRT == True:	
		tree = []
		path = []
		q_list = []
		v_list = []
		a_list =[]
		coeffs_l=[]
		new_path = []
		index = []

		index.append(-1)
		tree.append(q_start)
		new_node = q_start
		count = 0
		minspace = 0
		t = 1

		if not self.is_state_valid(new_node):
			count = 3000
		while not self.is_segment_valid(new_node,q_goal):
			if count>2000:
				break
			random_node = numpy.zeros(7)
			for i in range(7):
				random_node[i] = random.random()*(q_max[i]-q_min[i]) + q_min[i]
#				random_node[i] = random.random()*(q_goal[i]-q_start[i]) + q_start[i]
#				if random_node[i] > q_max[i]:
#					random_node[i] = q_max[i]
#				if random_node[i] < q_min[i]:
#					random_node[i] = q_min[i]

			minspace = numpy.linalg.norm(random_node-tree[0])
			closest_node = tree[0]
			minindex = 0
			for i in range(1,len(tree)):
				space = numpy.linalg.norm(random_node-tree[i])
				if space < minspace:
					minspace = space
					closest_node = tree[i]
					minindex = i
			
			new_node = closest_node+0.5*(random_node-closest_node)/minspace
			if self.is_state_valid(new_node):
				if self.is_segment_valid(closest_node,new_node):
                                	#new_node = closest_node+0.9*(new_node-closest_node)/numpy.linalg.norm(new_node-closest_node)
					tree.append(new_node)
					index.append(minindex)
					count = count + 1
					print "count:",count
					print "new node:",new_node
				else:
					new_node=q_start
			else:
				new_node=q_start
		
		#tree.append(q_goal)		
		if count>2000:
			if count==3000:
				print "invalid start node"
			q_list.append(q_start)
                	v_list,a_list,t = self.compute_simple_timing(q_list, 1)
			return q_list,v_list,a_list,t
		print "total count",count
		print "q_start",q_start
		print "q_goal",q_goal 

		i=len(tree)-1
		path.append(q_goal)	
		while not i==-1:
			path.insert(0,tree[i])
			i=index[i]	

        	#path = tree
		print "path",path
	#---------------------PRM-----------------------	
	else:   #get roadmap
		s_group=[]		
		s_group.append(q_start)
		print 'q-s', q_start
		g_group=[]
		g_group.append(q_goal)

		print 'q_g', q_goal
		connect = 0

		while connect == 0:
			accepte_s = 0
			accepte_g = 0
			k = 0
			i = 0
			#print "PRM_1"
			#generate one random sample 'sam' 	#with 7 D [1,2,3,4,5,6,7]
			sam = numpy.zeros(7)				
			for i in range(7):
				sam[i] = random.random()*(q_max[i]-q_min[i]) + q_min[i]
			#print "PRM_2"
			
			if self.is_state_valid(sam)==True:
				print "sample point",sam
				i = 0
				#print "PRM_3"
				while i < len(s_group) and accepte_s == 0:  #check if sam can see group_s
					if self.is_segment_valid(sam, s_group[i]) == True:
						accepte_s = 1
						#print "PRM_4"

						k=0
						while k < len(g_group) and accepte_g == 0:  #check if sam can see group_s
							#print "PRM_5"
							if self.is_segment_valid(sam, g_group[k]) == True: #'sam' can see g_group[i]
								accepte_g = 1
								#print "PRM_6"
							else: 
								accepte_g = 0
								connect=0
								k = k+1
								#print "PRM_7"
							#print "PRM_8"

						if accepte_g == 1:
							connect = 1
							#print "PRM_9"
						else:
							connect = 0
							#print "PRM_10"
						#print "PRM_11"

					else: 
						accepte_s = 0
						connect=0
						i = i+1
						#print "PRM_12"
		 
				if accepte_s == 1:  #if 'sam' can see s_group
					s_group.append(sam) #? s_group[] = s_group[]+sam
					print "s_group",s_group
					#print "PRM_13"
			
				else:
					k=0
					while k < len(g_group) and accepte_g == 0:  #check if sam can see group_g
						#print "PRM_14"
						if self.is_segment_valid(sam, g_group[k]) == True: #'sam' can see g_group[i]
							accepte_g = 1
							#print "PRM_15"
						else: 
							accepte_g = 0
							connect=0
							k = k+1
							#print "PRM_16"
					#print "PRM_17"
					if accepte_g == 1: # #if 'sam' can see g_group
						g_group.insert(0,sam) #? s_group[] =sam+g_group
						print "g_group",g_group
						#print "PRM_18"
					#print "PRM_19"
		
		
		pos_list = []		
		pos_list=pos_list+s_group
		pos_list=pos_list+g_group	#put s_group and g_group together	
		 
		print "q_start",q_start
		print "q_goal",q_goal
		print "pos_list",pos_list
		print "pos_list length",len(pos_list)
		#print "PRM_20"

	#-------------part2---------------------------		
		
		#finish matix PRM()[][]
		#point_name[] = ( 1,2,3,4)
		#pos_list[] = ([1234567],[1234567],[...]) #this will be the location of each point
		#value[] = (&,&,&,&,&) #this will be the value number of each point
		point_name = range(len(pos_list))
		#print "part2_1"
		m=0
		value=[]
		for i in range(len(pos_list)):
			value.append(999)
		value[0] = 0	
		visited = [] #generate 'visited node'
		current_point = pos_list[0]
		current_point_number = 0
		unvisited = point_name #point_name[]-starting_0  #(1,2,3,4....)
		unvisited[0] = None 
		#print "part2_2"
		kill = 0
		count_2=0
		while current_point_number != point_name[len(point_name)-1]: #and kill != 1: #(goal) if not connected to goal
			#print "part2_3"
			visited.append(current_point_number) #add point to visited
			#set number for nearby point
			#value = 'new point value' + 'new distance' -----record to matix, replace in (both way)
			#print "part2_4"
			for i in range(0,len(unvisited)):
				#print "part2_4.5"
				if (unvisited[i] != None) and (self.is_segment_valid(pos_list[current_point_number], pos_list[i]) == True):
					#print "part2_5"
					value[i] = min(value[i],(value[current_point_number] + numpy.linalg.norm(pos_list[i]-pos_list[current_point_number])))
					#print "part2_6"

			#new_point_number = unvisited.index(min(value))) #find smallest value[..] (within unvisited)
			minindex1=0
			#minindex1 = point_name[len(point_name)-1]
			minvalue=1000
			for i in range(1,len(unvisited)):
			
				#print "part2_6.1"
				if unvisited[i]!=None:
					#print "part2_6.2"
					if value[unvisited[i]]<minvalue:
						minvalue=value[unvisited[i]]
						minindex1=unvisited[i]
						#print "part2_6.5"
			current_point_number = minindex1
			unvisited[current_point_number] = None #get rid of the new_point_number from unvisited
			print "visited",visited
			print "unvisited",unvisited
			#print "part2_7"
			print 'current_point_number#', current_point_number
			#print 'point_name[len(point_name)-1]', point_name[len(point_name)-1]
			count_2=count_2+1
			print "count_2",count_2
			print "This process will be slow, please be patient."
			if visited[(len(visited)-1)]==0 and len(visited)>1:
				break
		
		#print "part2_7.5"
		print "value",value
		print "visited",visited
		print "q_start",q_start
		print "q_goal",q_goal
		print "pos_list",pos_list
		print "pos_list length",len(pos_list)
		print "-------------while loop end-------------------------------"


		#we will get:  value[] = (0,134,245,3,422....) #this will be the value number of each point
		#next step:depend on value[], find the path'''
		#known	#point_name = [0,1,2,3,4.....n]
		#new_point_name = [n,n-1,....,4,3,2,1]'''
		#visited = [0,5,8,76,...g]
		#value = (0,134,245,3,422....)
		#pos_list([1234567],[1234567],[...])
		print "value",value
		print "pos_list",pos_list
		
		path1=[]
		path1.append(pos_list[len(pos_list)-1])

		current_point1=pos_list[len(pos_list)-1]
		pos_list[len(pos_list)-1]=None

		
		current_index=0
		pos_temp=[]
		index_temp=[]
		min_index=0

		temp_value=0
		while 1:
			pos_temp=[]
			index_temp=[]
			for i in range(len(pos_list)):
				if pos_list[i]!=None:				
					if self.is_segment_valid(current_point1,pos_list[i]):
						pos_temp.append(pos_list[i])
						index_temp.append(i)

			temp_value = value[index_temp[0]]
			min_index=index_temp[0]
			for j in range(len(index_temp)):
				if value[index_temp[j]]<temp_value:
					temp_value=value[index_temp[j]]
					min_index=index_temp[j]

			print "min_index",min_index
			path1.insert(0,pos_list[min_index])
			current_point1=pos_list[min_index]
			pos_list[min_index]=None
			
			if(min_index==0):
				break
		#path1.insert(q_start)
		#print "xie wan le"
		print "path1",path1
		print "-------------------------------------------------------"
		path=path1
		
	#-----------------------shortcut--------------------------------------	
	
	new_path.append(path[0])
	shortcut_start = path[0]
	i = 0
	while i<(len(path)-1):
		i = i+1			
		if (not self.is_segment_valid(shortcut_start,path[i])):
			new_path.append(path[i-1])
			shortcut_start = path[i-1]
			print "shortcut node:",path[i-1]
	new_path.append(q_goal)
	print "shortcut done"
	print "shortcut path",new_path
	#-----------------------resample--------------------------------------
	i = 0	
	t_i=len(new_path)-1
	resample_path = []
	j=0
	n=0
	if numpy.linalg.norm(q_goal-q_start)>0.5:
		while (i < t_i): 
			while(n<1):
				#if j==t_i:
				#	for e in range(j):
				#		resample_path.append(path[i])
				#	break
				j=j+1
				a=new_path[i]
				b=new_path[i+j]
			
				seg = b-a
				seg_len = numpy.linalg.norm(seg)
				n = int(seg_len/0.5)
				if n==0:
					break;

			
			#if j==t_i:
			#	break
			resample_path.append(a)
		
			if n>1:
				for k in range(1,n):
					insert_point = a + k*seg/n
					resample_path.append(insert_point)
					print "insert point:",insert_point

			i=i+j
			j=0
			n=0
		resample_path.append(q_goal)
	else:
		resample_path=new_path		
	print "insert points done"
	print "resample path",resample_path
	print "resample path length",len(resample_path)
	#-----------------------trajectory computation----------------------------------	
	coeffs_l,v_list,a_list,t = self.compute_simple_timing_own2(resample_path,1)
        #v_list,a_list,t=self.compute_simple_timing(resample_path,1)
	q_list = resample_path
	#return q_list,v_list,a_list,t	
	#-----------------------trajectory plot--------------------------------------	
	coeffs_l4=[]	
	for i in range(4*len(q_list)-4):
		coeffs_l4.append(coeffs_l[4][i])
		#coeffs_l4.append(coeffs_l[i][4])	
	print "coeffs list",coeffs_l4
	print "coeffs list length",len(coeffs_l4)
	#print v_list
	#print a_list
	if self.show_plots == True:
		self.plot_trajectory(len(resample_path)-1, coeffs_l4, 1)
	return q_list,v_list,a_list,t
        # ---------------------------------------------------------------

    def project_plan(self, q_start, q_goal, q_min, q_max):
        q_list, v_list, a_list, t = self.motion_plan(q_start, q_goal, q_min, q_max)
        joint_trajectory = self.create_trajectory(q_list, v_list, a_list, t)
        return joint_trajectory

    def moveit_plan(self, q_start, q_goal, q_min, q_max):
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(q_goal)
        plan=self.group.plan()
        joint_trajectory = plan.joint_trajectory
        for i in range(0,len(joint_trajectory.points)):
            joint_trajectory.points[i].time_from_start = \
              rospy.Duration(joint_trajectory.points[i].time_from_start)
        return joint_trajectory        

    def create_trajectory(self, q_list, v_list, a_list, t):
        joint_trajectory = trajectory_msgs.msg.JointTrajectory()
        for i in range(0, len(q_list)):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = list(q_list[i])
            point.velocities = list(v_list[i])
            point.accelerations = list(a_list[i])
            point.time_from_start = rospy.Duration(t[i])
            joint_trajectory.points.append(point)
        joint_trajectory.joint_names = self.joint_names
        return joint_trajectory

    def execute(self, joint_trajectory):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory = joint_trajectory
        goal.goal_time_tolerance = rospy.Duration(0.0)
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

    def sample_polynomial(self, coeffs, i, T):
        q = coeffs[4*i+0]*T*T*T + coeffs[4*i+1]*T*T + coeffs[4*i+2]*T + coeffs[4*i+3]
        v = coeffs[4*i+0]*3*T*T + coeffs[4*i+1]*2*T + coeffs[4*i+2]
        a = coeffs[4*i+0]*6*T   + coeffs[4*i+1]*2
        return (q,a,v)

    def plot_series(self, t_vec, y_vec, title):
        fig, ax = plt.subplots()
        line, = ax.plot(numpy.random.rand(10))
        ax.set_xlim(0, t_vec[-1])
        ax.set_ylim(min(y_vec),max(y_vec))
        line.set_xdata(deepcopy(t_vec))
        line.set_ydata(deepcopy(y_vec))
        fig.suptitle(title)

    def move_arm_cb(self, feedback):
        print 'Moving the arm'
        self.mutex.acquire()
        q_start = self.q_current
        T = convert_from_message(feedback.pose)
        print "Solving IK"
        q_goal = self.IK(T)
        if len(q_goal)==0:
            print "IK failed, aborting"
            self.mutex.release()
            return

        print "IK solved, planning"
        q_start = numpy.array(self.q_from_joint_state(self.joint_state))
        trajectory = self.project_plan(q_start, q_goal, self.q_min, self.q_max)
        if not trajectory.points:
            print "Motion plan failed, aborting"
        else:
            print "Trajectory received with " + str(len(trajectory.points)) + " points"
            self.execute(trajectory)
        self.mutex.release()

    def no_obs_cb(self, feedback):
        print 'Removing all obstacles'
        self.scene.remove_world_object("obs1")
        self.scene.remove_world_object("obs2")
        self.scene.remove_world_object("obs3")
        self.scene.remove_world_object("obs4")

    def simple_obs_cb(self, feedback):
        print 'Adding simple obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)

        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.5, 0.5, 0)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,1))

    def complex_obs_cb(self, feedback):
        print 'Adding hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))

    def super_obs_cb(self, feedback):
        print 'Adding super hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.0, 0.2)) )
        self.scene.add_box("obs3", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.1)) )
        self.scene.add_box("obs4", pose_stamped,(0.1,0.5,0.1))


    def plot_cb(self,feedback):
        handle = feedback.menu_entry_id
        state = self.menu_handler.getCheckState( handle )
        if state == MenuHandler.CHECKED: 
            self.show_plots = False
            print "Not showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.UNCHECKED )
        else:
            self.show_plots = True
            print "Showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.CHECKED )
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()
        
    def joint_states_callback(self, joint_state):
        self.mutex.acquire()
        self.q_current = joint_state.position
        self.joint_state = joint_state
        self.mutex.release()

    def init_marker(self):

        self.server = InteractiveMarkerServer("control_markers")

        control_marker = InteractiveMarker()
        control_marker.header.frame_id = "/base"
        control_marker.name = "move_arm_marker"

        move_control = InteractiveMarkerControl()
        move_control.name = "move_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_y"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_z"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)

        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_y"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_z"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.BUTTON
        menu_control.always_visible = True
        box = Marker()        
        box.type = Marker.CUBE
        box.scale.x = 0.15
        box.scale.y = 0.03
        box.scale.z = 0.03
        box.color.r = 0.5
        box.color.g = 0.5
        box.color.b = 0.5
        box.color.a = 1.0
        menu_control.markers.append(box)
        box2 = deepcopy(box)
        box2.scale.x = 0.03
        box2.scale.z = 0.1
        box2.pose.position.z=0.05
        menu_control.markers.append(box2)
        control_marker.controls.append(menu_control)

        control_marker.scale = 0.25        
        self.server.insert(control_marker, self.control_marker_feedback)

        self.menu_handler = MenuHandler()
        self.menu_handler.insert("Move Arm", callback=self.move_arm_cb)
        obs_entry = self.menu_handler.insert("Obstacles")
        self.menu_handler.insert("No Obstacle", callback=self.no_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Simple Obstacle", callback=self.simple_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Hard Obstacle", callback=self.complex_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Super-hard Obstacle", callback=self.super_obs_cb, parent=obs_entry)
        options_entry = self.menu_handler.insert("Options")
        self.plot_entry = self.menu_handler.insert("Plot trajectory", parent=options_entry,
                                                     callback = self.plot_cb)
        self.menu_handler.setCheckState(self.plot_entry, MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, "move_arm_marker",)

        self.server.applyChanges()

        Ttrans = tf.transformations.translation_matrix((0.6,0.2,0.2))
        Rtrans = tf.transformations.rotation_matrix(3.14159,(1,0,0))
        self.server.setPose("move_arm_marker", convert_to_message(numpy.dot(Ttrans,Rtrans)))
        self.server.applyChanges()


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()
