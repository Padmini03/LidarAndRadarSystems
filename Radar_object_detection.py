#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # visualize point clouds
import cv2
import math
import os
import pandas as pd


record_number = 3 # Either record 1 or record 2

root = f"record3/record{record_number}/"
path_blickfeld = f"{root}/blickfeld"
path_camera = f"{root}/camera"
path_groundtruth = f"record1/groundtruth/"
path_radar = f"{root}/radar"
path_velodyne = f"{root}/velodyne"



def readPC(path):
    """
       Will return a numpy array of
       shape:
       Nx4 for LiDAR Data (x,y,z,itensity)
       Nx5 for RADAR Data (x,y,z,velocity,itensity)
       
    """
    return np.loadtxt(path)

def readImage(path):
    """
       Will return an numpy array of
       shape height x width x 3.
    """
    return cv2.imread(path) #[:,:,[2,1,0]]   #### cv2.imread() method loads an image from the specified file

def readLabels(path):
    """
       Reads the ground truth labels.
       In the labels are the following
       information stored:
       1. width in m
       2. length in m
       3. height in m
       4.-6. Coordinates of the center in m
       7. yaw rotation in degree
    """
    return np.loadtxt(path)


# In[4]:


frame_id = 4
assert (record_number == 1 and frame_id < 240) or (record_number == 2 and frame_id < 100) or (record_number == 3 and frame_id < 295),      "Record number 1 only has 240 frames and"    " record number 2 only has 100 frames."
pc_blick = readPC(f"{path_blickfeld}/{frame_id:06d}.csv")
print(f"{path_blickfeld}/{frame_id:06d}.csv")

pc_velo = readPC(f"{path_velodyne}/{frame_id:06d}.csv")
print(f"{path_velodyne}/{frame_id:06d}.csv")

pc_radar = readPC(f"{path_radar}/{frame_id:06d}.csv")
print(f"{path_radar}/{frame_id:06d}.csv")

label = readLabels(f"{path_groundtruth}/{frame_id:06d}.csv")
print(f"{path_groundtruth}/{frame_id:06d}.csv")

img = readImage(f"{path_camera}/{frame_id:06d}.jpg")
print(f"{path_camera}/{frame_id:06d}.jpg")


# In[5]:


### To get the dimensions of the array

print(f"Blickfeld point cloud shape: {pc_blick.shape}\nVelodyne point cloud shape: {pc_velo.shape}\nRadar point cloud shape: {pc_radar.shape}\nImage shape: {img.shape}")  ### The output gives the dimensions variable to get width, height and number of channels for each pixel.

### width : 720
### height : 1280
### No of channels per pixel : 3


# In[6]:


pc_blick


# In[7]:


pc_velo


# In[8]:


pc_radar


# In[9]:


#label


# In[10]:


img

# In[11]:


plt.imshow(img) # cv2 color order is (blue, green, red) 
                # and of matplotlib.pyplot is (red, green, blue)


# In[12]:


def readImage(path):
    """
       Will return an numpy array of
       shape height x width x 3.
    """
    return cv2.imread(path)[:,:,[2,1,0]]    ### cv2.imread() method loads an image from the specified path


# In[13]:


img = readImage(f"{path_camera}/{frame_id:06d}.jpg")
print(f"{path_camera}/{frame_id:06d}.jpg")


# In[14]:


plt.imshow(img)


# In[15]:


plt.figure(figsize=(40,30)) # Size of the Figure
plt.axis("off") # Deletes the axis
plt.imshow(img) 


# Can use matplotlib.pyplot.scatter to visualize the point clouds, but only 2D.

# In[16]:


plt.scatter(pc_blick[:,0],pc_blick[:,1], s = 1, c = pc_blick[:,2])


# For 3D we can use plotly.graph_objects:

# In[17]:


data = [go.Scatter3d(x = pc_blick[:,0],
                     y = pc_blick[:,1],
                     z = pc_blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_blick[:,2],
                        'colorscale':'rainbow',
})
]

go.Figure(data=data)


# In[18]:


data = [go.Scatter3d(x = pc_velo[:,0],
                     y = pc_velo[:,1],
                     z = pc_velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_velo[:,2],
                        'colorscale':'rainbow',
})
]
go.Figure(data=data)


# In[19]:


data = [go.Scatter3d(x = pc_radar[:,0],
                     y = pc_radar[:,1],
                     z = pc_radar[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_radar.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_radar[:,2],
                        'colorscale':'rainbow',
})
]

go.Figure(data=data)


# In[20]:


data = [go.Scatter3d(x = pc_blick[:,0],
                     y = pc_blick[:,1],
                     z = pc_blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_blick[:,2],
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20]},
        'yaxis': {'range': [0, 40]},
        'zaxis': {'range': [-20., 20.]},
    }
                        
)
go.Figure(data=data,layout=layout)


# In[21]:


data = [go.Scatter3d(x = pc_velo[:,0],
                     y = pc_velo[:,1],
                     z = pc_velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_velo[:,2],
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20]},
        'yaxis': {'range': [0, 40]},
        'zaxis': {'range': [-20., 20.]},
    }
                        
)
go.Figure(data=data,layout=layout)


# In[22]:


data = [go.Scatter3d(x = pc_radar[:,0],
                     y = pc_radar[:,1],
                     z = pc_radar[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_radar.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_radar[:,2],
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20]},
        'yaxis': {'range': [0, 40]},
        'zaxis': {'range': [-20., 20.]},
    }
                        
)
go.Figure(data=data,layout=layout)


# In[23]:


data = [go.Scatter3d(x = pc_blick[:,0],
                     y = pc_blick[:,1],
                     z = pc_blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
                go.Scatter3d(x = pc_velo[:,0],
                    y = pc_velo[:,1],
                    z = pc_velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "yellow",
                        'colorscale':'rainbow',
}),     
                go.Scatter3d(x = pc_radar[:,0],
                    y = pc_radar[:,1],
                    z = pc_radar[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_radar.shape[0]),
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
})
]
layout = go.Layout(
    scene={
        'xaxis': {'range': [-20, 20], 'rangemode': 'tozero', 'tick0': -5},
        'yaxis': {'range': [0, 40], 'rangemode': 'tozero', 'tick0': -5},
        'zaxis': {'range': [-20., 20.], 'rangemode': 'tozero'}
    }
)
go.Figure(data=data, layout = layout)


# In[24]:


def get_dynamic_points(pc_radar):
    rad_vel=pc_radar[:,3]
    dynamic_arr = rad_vel[np.nonzero(rad_vel)]
    #print(dynamic_arr)
    x = pc_radar[:,0]
    y = pc_radar[:,1]

    x_axis = []
    y_axis = []

    dynamic_ind = np.nonzero(rad_vel)
    dyn_len = len(dynamic_ind)
    #print("Indices of non zero elements : ", dynamic_ind)
    #print("Number of dynamic elements : ", dyn_len)
    
    for i in dynamic_ind:
        x_axis.append(x[i])
        y_axis.append(y[i])

    #print("x-axis of dynamic points:",x_axis)
    #print("y-axis of dynamic points:",y_axis)
    return x_axis, y_axis


# In[26]:


x_radar_points=pc_radar[:, 0]
y_radar_points=pc_radar[:, 1]
plt.scatter(x_radar_points,y_radar_points)
plt.title("Graph using Radar Sensor")
plt.ylabel('y_radar_points')
plt.xlabel('x_radar_points')
plt.show()


# In[27]:


x_axis,y_axis = get_dynamic_points(pc_radar)
plt.title("Graph for Dynamic points of Radar Sensor")

plt.scatter(x_radar_points,y_radar_points, color='green')
plt.scatter(x_axis,y_axis, color='orange')

plt.ylabel('y_radar_points')
plt.xlabel('x_radar_points')
plt.show()


# In[28]:


cen_x = np.mean(x_axis)
cen_y = np.mean(y_axis)

x_axis,y_axis = get_dynamic_points(pc_radar)
plt.title("Centre of cluster using Radar Sensor")

plt.scatter(x_radar_points,y_radar_points, color='green')
plt.scatter(x_axis,y_axis, color='orange')
plt.scatter(cen_x, cen_y, color='red')

plt.ylabel('y_radar_points')
plt.xlabel('x_radar_points')
plt.show()


# In[49]:


def calculate_centre(x_axis, y_axis):
    cen_x = np.mean(x_axis)
    cen_y = np.mean(y_axis)

    print("centre of x-axis and y-axis is", cen_x,cen_y)
    
    dist = math.sqrt((cen_x*cen_x)+(cen_y*cen_y))
    #print(dist)
    return dist


# In[50]:


root = f"record3/record3/"
distance = []

for i in range(295):
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    #print(pc_radar)
    x_axis, y_axis = get_dynamic_points(pc_radar)
    
    dist = calculate_centre(x_axis, y_axis)
    distance.append(dist)
    print(i, dist)
    
#print(distance)
#print(len(distance))


# In[48]:


ID=[]
for i in range(295):
    ID.append(i)

print(ID)
plt.scatter(ID, distance, color='hotpink')
plt.title("Graph using Radar Sensor")
plt.ylabel('Distance[m]')
plt.xlabel('Image ID')
plt.show()


# In[ ]:




