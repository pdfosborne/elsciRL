# Sailing Simulator
# - https://github.com/topics/sailing-simulator
# - Simple sailing simulator from https://github.com/PPierzc/ai-learns-to-sail
#   - https://github.com/PPierzc/ai-learns-to-sail/blob/master/tasks/channel.py
import io
import numpy as np
import matplotlib.pyplot as plt
from elsciRL.examples.environments.sailing_image import SailingImageData

class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, local_setup_info:dict={}) -> None:
        """Initialize Engine"""
        #self.Environment = "Engine Initialization"
        self.x_limit = 10
        self.y_limit = local_setup_info["y_limit"]
        self.angle_limit = np.pi / 2
        self.supervised_rewards = local_setup_info["supervised_rewards"]
        # Precision parameter
        self.obs_precision = local_setup_info["obs_precision"]

        # Ledger of the environment with meta information for the problem
        ledger_required = {
            'id': 'Unique Problem ID',
            'type': 'Language/Numeric',
            'description': 'Problem Description',
            'goal': 'Goal Description'
            }
        
        ledger_optional = {
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year',
            'render_data':{'render_mode':'rgb_array', 
                           'render_fps':4}
        }
        ledger_gym_compatibility = {
            # Limited to discrete actions for now, set to arbitrary large number if uncertain
            'action_space_size':2, 
        }
        self.ledger = ledger_required | ledger_optional | ledger_gym_compatibility
        # Initialize history
        self.action_history = []
        self.obs_history = []
        
        

    # --------------------------
    # Defined functions used by engine source
    @staticmethod
    def vel(theta, theta_0=0, theta_dead=np.pi / 12):
        return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)
    
    @staticmethod
    def rew(theta, theta_0=0, theta_dead=np.pi / 12):
        return Engine.vel(theta, theta_0, theta_dead) * np.cos(theta)
    # --------------------------

    def reset(self, start_obs:str=None, render_dir:str=None):
        """Fully reset the environment."""
        # Allow reset to be at fixed start position or random
        if start_obs:
            self.x = np.round(float(start_obs.split('_')[0]),self.obs_precision)
            self.angle = np.round(float(start_obs.split('_')[1]),1)
        else:
            self.x = 0 #np.round(np.random.randint(-9.9, 9.9),4) # Changed to rand_int to reduce num of start states
            self.angle = 0  # always start with angle 0
        self.y = 0
        obs = "{n:.{d}f}".format(n=self.x, d=self.obs_precision)+'_'+"{:0.1f}".format(self.angle)
        
        if render_dir:
                
            # SHOW PRETTY IMAGE OF PROBLEM
            raw_image = SailingImageData['data'].split(",")

            width = 240
            height = 300

            full_array = []
            column_counter = 0
            row = []
            pixel_counter = 0
            pixel_list = []
            for input_item in raw_image:
                pixel_item = int(input_item.replace(" ",""))
                if pixel_counter == 3:
                    # new pixel and reset pixel counter
                    pixel_counter = 0
                    pixel_list = []
                    # Add 3-d pixel to row
                    if column_counter == width:
                        # Add row to full array
                        full_array.append(row)
                        # new row and reset column counter
                        column_counter = 0
                        row = []
                
                    row.append(pixel_list)
                    column_counter+=1
                
                pixel_list.append(pixel_item)
                pixel_counter+=1

            render = np.array(full_array)
            plt.imshow(render, interpolation='nearest')
            plt.axis('off')
            plt.title("Sailing Simulation \n Simple River with Fixed Wind Direction")
            plt.show()
            plt.pause(5)
            plt.savefig(render_dir,bbox_inches='tight')
            plt.close()
        
        return obs

    
    def step(self, state:any=None, action:any=None):
        """Enact an action."""
        self.action_history.append(action)
        a = [-0.1, 0.1][action]
        # Observation space
        self.x += np.round((Engine.vel(self.angle + a) * np.sin(self.angle + a)),self.obs_precision) # Round x to Ndp
        self.y += np.round((Engine.vel(self.angle + a) * np.cos(self.angle + a)),4) # Round y to 4dp
        self.angle = np.round(self.angle+a,1) 
        #obs = str(self.x)+'_'+str(self.angle)
        obs = "{n:.{d}f}".format(n=self.x, d=self.obs_precision)+'_'+"{:0.1f}".format(self.angle) # fix - https://docs.python.org/3.4/library/string.html#format-specification-mini-language
        self.obs_history.append(obs)
        # Reward signal
        # - Added flag for whether we give agent immediate positive reward
        # - Update: Added scale factor if using supervised rewards to not override goal rewards
        if self.supervised_rewards=="True":
            reward = Engine.rew(self.angle)/10
        else:
            reward = 0

        # Termination signal
        # - Source: Terminal only on hitting piers/walls, otherwise continues to action limit
        # - Update: Add terminal state if y > 25 (or another arbitrary value)
        # - Update: Limit angle to [-90,90] degrees (i.e. no backwards sailing)
        if np.abs(self.x)>self.x_limit:
            reward = -1
            terminated = True
        elif np.abs(self.y)>self.y_limit:
            reward = 1
            terminated = True
        elif np.abs(self.y)<0:
            reward = -1
            terminated = True
        elif np.abs(self.angle)>self.angle_limit:
            #print("\n \t - Angle limit reached")
            reward = -1
            terminated = True
        else:
            terminated = False
        
        return obs, reward, terminated, {}

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        # Action space: [0,1] for turn slightly left or right
        # - Kept as binary but might be better as continuous [-0.1, 0.1]
        legal_moves = [0, 1]
        return legal_moves
    
    def render(self, state:any=None):
        """Render the environment."""            
        #render = print("Current State: ", state, " | Action History: ", self.action_history)
        # state = x_angle
        x = self.x
        y = self.y
        angle = self.angle
        # Angle is bearing into wind -pi/2 < angle < pi/2
        if angle < np.pi/2:
            U = np.sin(angle)
            V = np.cos(angle)
        elif angle == np.pi/2:
            U = 1
            V = 0
        elif angle == -np.pi/2:
            U = -1
            V = 0
        else:
            U = np.sin(angle)
            V = -np.cos(angle)

        DPI = 128
        fig, ax = plt.subplots(figsize=(5,5), dpi = DPI)
        ax.scatter(x,y,c='b',marker='x',alpha=1)
        ax.quiver(x,y,U,V,angles='uv',scale_units='xy')
        if y > 1:
            ax.text(x+0.5,y-1,'Sailboat',color='b')

        # Draw wind direction
        ax.quiver(0,25,0,-1,angles='uv',scale_units='xy',color='r')
        ax.text(0,25.25,'Wind',color='r')


        ax.plot([10,10],[0,25],'r')
        ax.plot([-10,-10],[0,25],'r')
        ax.set_title("Sailboat Position with Direction against Wind")
        ax.set_xlabel("Horizontal Position (x)")
        ax.set_ylabel("Vertical Position (y)")
        # Save as rgba array 
        # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
        

        fig.canvas.draw()
        # data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # render = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        render = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return render

    def close(self):
        """Close/Exit the environment."""
        print("Environment Closed")