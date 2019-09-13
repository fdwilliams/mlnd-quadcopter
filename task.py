import numpy as np
from physics_sim import PhysicsSim

sim_min = np.array([-150, -150, 0])
sim_max = np.array([150, 150, 300])
sim_center = (sim_min + sim_max) / 2
box_size = np.array([100, 100, 0])

#scale for state position
pos_scale = 1 / np.array([300, 300, 300])

#task sphere radius
sphere_radius = 50

push_position = False #include position in state
push_angles = True #include angles in state
push_target = False #include target position in state
push_difference = True #include (target - position) in state
push_velocity = False #include velocity in state
push_angvel = False #include angular velocity in state
push_rotors = False #include rotor speeds in state

#if true, concat all action_repeat steps as one state
#otherwise, just output the last step's state
output_all_repeats = False

#converts radians to revolutions [-0.5..0.5]
def rad_to_rev(angles):
    #wrap
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    angles = angles / np.pi / 2
    return angles

def calc_reward(task):
    dr = distance_reward(task.sim.pose[:3], task.target_pos)
    tr = tilt_reward(task.sim.pose[3:])
    #jr = jerk_reward(task.cur_rotor_speeds, task.last_rotor_speeds, task.action_range)
    #vr = velocity_reward(task.sim.v)
    reward = dr * 1 + tr * 1 # + vr * 0.25 + jr * 0.1
    return reward

#smaller distance to target = better
def distance_reward(pos, target):
    distance = np.linalg.norm(pos - target)
    distance /= 150
    distance = 0.5-distance
    #distance = np.square(distance) * np.sign(distance)
    return distance

#slower = better
def velocity_reward(vel):
    vel = np.sqrt(np.square(vel).sum())
    vel /= 150
    vel = 0.5-vel
    #vel = np.square(vel) * np.sign(vel)
    return vel

#angles close to 0 = better
def tilt_reward(angles):
    """angles are assumed to be within [0..2pi]"""
    angles = rad_to_rev(angles)
    angles = np.abs(angles[:2]).sum() #punish for tilting, [-1..0]
    angles = 0.5-angles
    #angles = np.square(angles) * np.sign(angles)
    return angles

#small difference in rotor speeds = better
def jerk_reward(cur_rotors, prev_rotors, action_range):
    """punish for changing rotor speeds quickly"""
    #difference in raw speeds
    #sometimes these arent ndarrays.. (in the case of provided ddpg code)
    reward = np.abs(np.array(cur_rotors) - np.array(prev_rotors))
    reward /= action_range
    reward = 0.5-np.mean(reward)
    #reward = np.square(reward) * np.sign(reward)
    return reward

class BaseTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, runtime=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        #start position (no angles)
        self.start_pos = np.zeros(3)
        #target position
        self.target_pos = np.zeros(3)

        # Simulation
        self.sim = PhysicsSim(runtime = runtime)
        self.action_repeat = 3
        
        self.action_size = 4
        speed_hover = 400 #speed required to hover in place when angles are 0
        speed_range = 400 #allowed rotor speed range centered around speed_hover
        self.action_low = speed_hover - speed_range / 2
        self.action_high = speed_hover + speed_range / 2
        self.action_range = self.action_high - self.action_low
        
        #keep track of previous rotor speeds (for jerkiness reward function)
        self.cur_rotor_speeds = np.zeros(self.action_size)
        self.last_rotor_speeds = np.zeros(self.action_size)
        
        self.state_size = self.get_state_frame().shape[0] * \
            (self.action_repeat if output_all_repeats else 1)
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        return calc_reward(self)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        states = []
        done = False
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            states.append(self.get_state_frame())
        #if output_all_repeats, then concatenate all states into one vector, else send the last state
        next_state = np.concatenate(states) if output_all_repeats else states[-1]
        #update stored rotor speeds
        self.last_rotor_speeds = self.cur_rotor_speeds
        self.cur_rotor_speeds = rotor_speeds
        return next_state, reward, done

    def get_state_frame(self):
        """Returns one frame of state inputs
            (position, angles, velocity, anglular velocity, target position)
        """
        state = []
        #angles/angular velocity is pushed as revolutions
        if push_position:
            state.append(self.sim.pose[:3] * pos_scale)
        if push_angles:
            state.append(rad_to_rev(self.sim.pose[3:]))
        if push_velocity:
            state.append(self.sim.v * pos_scale)
        if push_angvel:
            state.append(rad_to_rev(self.sim.angular_v))
        if push_target:
            state.append(self.target_pos * pos_scale)
        if push_difference:
            state.append((self.target_pos - self.sim.pose[:3]) / pos_scale) #add difference between target and position
        if push_rotors:
            state.append((np.array(self.cur_rotor_speeds) - self.action_low) / self.action_range) #0..1 scaled rotor speed
        return np.concatenate(state)

    def get_init_pose(self):
        """Returns initial pose (start pos + euler 0s)"""
        return np.concatenate([self.start_pos, np.zeros(3)])

    def reset(self):
        """Reset the sim to start a new episode."""
        #manually reset init_pose
        self.last_rotor_speeds = np.zeros(self.action_size) #reset rotor speeds
        self.sim.init_pose = self.get_init_pose()
        self.sim.reset()
        state = np.concatenate([self.get_state_frame()] * (self.action_repeat if output_all_repeats else 1))
        return state

    
def point_on_sphere(radius):
    p = np.random.randn(3)
    p /= np.linalg.norm(p)
    p *= radius
    return p

class HoverTask(BaseTask):
    def __init__(self, runtime=5.):
        super(HoverTask, self).__init__(runtime)
    def reset(self):
        #center start position in the simulation
        self.start_pos = sim_center + point_on_sphere(sphere_radius)
        #set target equal to start position
        self.target_pos = self.start_pos.copy()
        return super().reset()

class TravelTask(BaseTask):
    def __init__(self, runtime=5.):
        super(TravelTask, self).__init__(runtime)

    def reset(self):
        #set start to a random position around the center
        self.start_pos = sim_center + point_on_sphere(sphere_radius)
        #set target to a random position around the center
        self.target_pos = sim_center + point_on_sphere(sphere_radius)
        return super().reset()
