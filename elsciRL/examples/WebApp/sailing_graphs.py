import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@staticmethod
def vel(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)

@staticmethod
def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)


class Analysis:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def render(self, state:any=None):
        """Render the environment.
        Args:
            state[x,y,angle]: The current state of the environment.
                - x: The horizontal position of the sailboat.
                - y: The vertical position of the sailboat.
                - angle: The angle of the sailboat.
            Returns:
                render: The rendered environment."""            
        x = float(state.split('_')[0])
        y = 5 # Not output by environment so using dummy value for display
        angle = float(state.split('_')[1])
        print("PLOT DATA = ", x, y, angle)
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
        ax.set_xlabel(f"Horizontal Position ({x})")
        ax.set_ylabel(f"Vertical Position ({y})")
        # Save as rgba array 
        # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
        return fig
    
    def trace_plot(self, experiments_names:list=['']):
        """Define experiment directories to run and override names for
        improved plot titles."""
        path = self.save_dir 
        path_folders = os.listdir(path)
        for n,folder in enumerate(path_folders):
            if os.path.isdir(path+'/'+folder):
                if (folder=='Supervised_Instr_Experiment')|(folder=='Standard_Experiment'):
                    exp_path = path + '/' + folder
                exp_path_folders = os.listdir(exp_path)

                count_check = 0
                policy_list = []
                for result_folders in exp_path_folders:
                    if os.path.isdir(exp_path+'/'+result_folders):
                        if 'testing' in result_folders:
                            testing_results_path = exp_path + '/' + result_folders
                            path_csv=glob.glob(testing_results_path+"/*.csv")
                            results = pd.read_csv(path_csv[0])
                            policy = results['action_history'].mode()[0]
                            policy_fix = policy.split(',')
                            policy_fix = [int(i.replace('[1','1').replace('[0','0').replace('1]','1').replace('0]','0')) for i in policy_fix]
                            policy_list.append(policy_fix)
                            count_check += 1
                #print(count_check)
                # Re-applies actions made by agent to observe path
                if experiments_names[n]!='':
                    exp_title = experiments_names[n]
                else:
                    exp_title = folder
                
                plt.scatter(0,0,marker='x', color='b')
                training_policies = policy_list
                for action_list in training_policies:
                    x = 0
                    y = 0
                    angle = 0
                    x_list=[]
                    y_list=[]
                    for action in action_list:
                        a = [-0.1,0.1][action]
                        
                        #print(x,"|", y, "|", angle)
                        x += np.round((vel(angle + a) * np.sin(angle + a)),4) # Round x to 2dp
                        y += np.round((vel(angle + a) * np.cos(angle + a)),4) # Round y to 2dp
                        angle=np.around(angle+a,1)
                        # if (x > 3)&(x<5):
                        #     print("---")
                        #     print("{:0.4f}".format(x),"|", "{:0.4f}".format(y), "|", "{:0.1f}".format(angle))
                        x_list.append(x)
                        y_list.append(y)

                    if np.abs(x_list[-1])>=10:
                        plt.plot(x_list,y_list,'r',alpha=0.75)
                    elif np.abs(y_list[-1])>=24:
                        plt.plot(x_list,y_list,'g',alpha=0.75)
                    elif np.abs(y_list[-1])<0:
                        plt.plot(x_list,y_list,'r',alpha=0.75)
                    else:
                        plt.plot(x_list,y_list,'k',alpha=0.75)
                        
                    plt.scatter(x_list[-1],y_list[-1],marker='x', color='r')
                    plt.plot([10,10],[0,25],'r')
                    plt.plot([-10,-10],[0,25],'r')
                    plt.title(exp_title + "\n Sailboat Path for each Trained Agent's Output Policy")
                    plt.xlabel("Horizontal Position (x)")
                    plt.ylabel("Vertical Position (y)")

                save_path = os.path.join(path, folder, 'trace_plot.png')

                plt.savefig(save_path)
                plt.show()
                plt.close()            
                #plt.show()

