#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install opencv-python')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # visualize point clouds
import cv2
import math

record_number = 1    # Either record 1 or record 2

root = f"Dataset 2022{record_number}/"
path_blickfeld = f"Dataset 2022{record_number}/blickfeld"
path_camera = f"Dataset 2022{record_number}/camera"
path_groundtruth = f"Dataset 2022{record_number}/groundtruth"
path_radar = f"Dataset 2022{record_number}/radar"
path_velodyne = f"Dataset 2022{record_number}/velodyne"


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
    return cv2.imread(path)#[:,:,[2,1,0]]

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


# In[5]:


frame_id = 62
assert (record_number == 1 and frame_id < 240) or (record_number == 2 and frame_id < 100),     "Record number 1 only has 240 frames and"    " record number 2 only has 100 frames."
pc_blick = readPC(f"{path_blickfeld}/{frame_id:06d}.csv")
pc_velo = readPC(f"{path_velodyne}/{frame_id:06d}.csv")
pc_radar = readPC(f"{path_radar}/{frame_id:06d}.csv")
label = readLabels(f"{path_groundtruth}/{frame_id:06d}.csv")
img = readImage(f"{path_camera}/{frame_id:06d}.jpg")


# In[6]:


print(f"Blickfeld point cloud shape: {pc_blick.shape}\nVelodyne point cloud shape: {pc_velo.shape}\nRadar point cloud shape: {pc_radar.shape}\nLabel data shape: {label.shape}\nImage shape: {img.shape}")


# In[7]:


pc_blick


# In[8]:


pc_velo


# In[9]:


pc_radar


# In[10]:


label


# In[11]:


img


# ## Visualization
# To understand the data easier, we visualize them. We can use the package matplotlib.pyplot and/or plotly.

# In[12]:


plt.imshow(img) # cv2 color order is (blue, green, red) 
                # and of matplotlib.pyplot is (red, green, blue)


# In[13]:


def readImage(path):
    """
       Will return an numpy array of
       shape height x width x 3.
    """
    return cv2.imread(path)[:,:,[2,1,0]]


# In[14]:


img = readImage(f"{path_camera}/{frame_id:06d}.jpg")


# In[15]:


plt.imshow(img)


# In[16]:


plt.figure(figsize=(40,30)) # Size of the Figure
plt.axis("off") # Deletes the axis
plt.imshow(img) 


# Can use matplotlib.pyplot.scatter to visualize the point clouds, but only 2D.

# In[17]:


plt.scatter(pc_blick[:,0],pc_blick[:,1], s = 1, c = pc_blick[:,2])


# For 3D we can use plotly.graph_objects:

# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


data = [go.Scatter3d(x = pc_blick[:,0],
                     y = pc_blick[:,1],
                     z = pc_blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_blick[:,2],
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = [label[3]],
                     y = [label[4]],
                     z = [label[5]],
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 5,
                        'color': "red",
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


# In[26]:


def rt_matrix(roll=0, pitch=0, yaw=0):
    """
        Returns a 3x3 Rotation Matrix. Angels in degree!
    """
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    
    # Rotationmatrix
    rot = np.dot(np.dot(np.array([[c_y, - s_y,   0],
                                  [s_y,   c_y,   0],
                                  [0,      0,    1]]),
                        np.array([[c_p,    0,    s_p],
                                  [0,      1,    0],
                                  [-s_p,   0,    c_p]])),
                        np.array([[1,      0,    0],
                                  [0,     c_r, - s_r],
                                  [0,     s_r,   c_r]]))
    return rot

def rotate_points(points, rot_t):
    """
        Input must be of shape N x 3
        Returns the rotated point cloud for a given roation matrix 
        and point cloud.
    """
    points[0:3,:] = np.dot(rot_t, points[0:3,:])
    return points

def make_boundingbox(label):
    """
        Returns the corners of a bounding box from a label.
    """
    corner = np.array([
        [+ label[0]/2, + label[1]/2, + label[2]/2],
        [+ label[0]/2, + label[1]/2, - label[2]/2],
        [+ label[0]/2, - label[1]/2, + label[2]/2],
        [+ label[0]/2, - label[1]/2, - label[2]/2],
        [- label[0]/2, + label[1]/2, + label[2]/2],
        [- label[0]/2, - label[1]/2, + label[2]/2],
        [- label[0]/2, + label[1]/2, - label[2]/2],
        [- label[0]/2, - label[1]/2, - label[2]/2],
    ])
    corner = rotate_points(corner, rt_matrix(yaw = label[6]))
    corner = corner + label[3:6]
    return corner


# In[27]:


bb = make_boundingbox(label)

# New bounding box for the visualization
bb = np.array([bb[0],bb[1],bb[3],bb[2],bb[0],bb[4],bb[5],
               bb[2],bb[3],bb[7],bb[5],bb[4],bb[6],bb[7],bb[6],bb[1]])


# In[28]:


data = [go.Scatter3d(x = pc_blick[:,0],
                     y = pc_blick[:,1],
                     z = pc_blick[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_blick.shape[0]),
                    marker={
                        'size': 2,
                        'color': pc_blick[:,2],
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = [label[3]],
                     y = [label[4]],
                     z = [label[5]],
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 5,
                        'color': "red",
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


# In[29]:


x_min=bb.min(0)
print(pc_blick[:,0:3])
print(pc_blick[0,0])


# In[30]:


x_max=bb.max(0)
print(x_max)
print(x_max[0:1])
print(label)


# In[31]:


x_min=bb.min(0)
x_max=bb.max(0)
a=[]
for i in range(pc_blick.shape[0]):
    if (pc_blick[i,0]>=x_min[0] and pc_blick[i,0]<=x_max[0]):
        if (pc_blick[i,1]>=x_min[1] and pc_blick[i,1]<=x_max[1]):
            if (pc_blick[i,2]>=x_min[2] and pc_blick[i,2]<=x_max[2]):
                a.append(pc_blick[i])
a=np.array(a)
print(a)
#print(a[1][:,0])


# In[32]:


x_min=bb.min(0)
x_max=bb.max(0)
count=0
a=[]
for i in range(pc_velo.shape[0]):
    if (pc_velo[i,0]>=x_min[0] and pc_velo[i,0]<=x_max[0]):
        if (pc_velo[i,1]>=x_min[1] and pc_velo[i,1]<=x_max[1]):
            if (pc_velo[i,2]>=x_min[2] and pc_velo[i,2]<=x_max[2]):
                a.append(pc_velo[i])
                count=count+1
a=np.array(a)
print(a)


# In[33]:


def get_sub_points_of_object(obj,pc):
    """
       Determines those points of the point cloud (pc)
       that are inside of the object (obj).
    """
    # 1. Get the bounding box corners.
    bb1=make_boundingbox(obj)
    x_min=bb1.min(0)
    x_max=bb1.max(0)
    
    #count=0
    a=[]
    for i in range(pc.shape[0]):
        if (pc[i,0]>=x_min[0] and pc[i,0]<=x_max[0]):
            if (pc[i,1]>=x_min[1] and pc[i,1]<=x_max[1]):
                if (pc[i,2]>=x_min[2] and pc[i,2]<=x_max[2]):
                    a.append(pc[i])
    a=np.array(a)
    return a
    
# In[34]:


def distance_of_center_points(obj):
    distance_of_the_object=math.sqrt((obj[3]*obj[3])+(obj[4]*obj[4])+(obj[5]*obj[5]))
    return distance_of_the_object
    print(distance_of_the_object)


# In[35]:


subpoints = get_sub_points_of_object(label, pc_blick[:,0:3])
#print(subpoints)
count=len(subpoints)
print(count)


# In[36]:


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
        go.Scatter3d(x = subpoints[:,0],
                     y = subpoints[:,1],
                     z = subpoints[:,2],
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                    mode='lines', type='scatter3d',
                    line={
                        'width': 10,
                        'color': "red",
                        'colorscale':'rainbow'
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


# In[37]:


print(len(subpoints))
subpoints = get_sub_points_of_object(label, pc_velo[:,0:3])
count=len(subpoints)
print(count)


# In[38]:


data = [go.Scatter3d(x = pc_velo[:,0],
                     y = pc_velo[:,1],
                     z = pc_velo[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_velo.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = subpoints[:,0],
                     y = subpoints[:,1],
                     z = subpoints[:,2],
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                    mode='lines', type='scatter3d',
                    line={
                        'width': 10,
                        'color': "red",
                        'colorscale':'rainbow'
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


# In[39]:


print(len(subpoints))
subpoints = get_sub_points_of_object(label, pc_radar[:,0:3])
print(len(subpoints))


# In[40]:


data = [go.Scatter3d(x = pc_radar[:,0],
                     y = pc_radar[:,1],
                     z = pc_radar[:,2],
                    mode='markers', type='scatter3d',
                    text=np.arange(pc_radar.shape[0]),
                    marker={
                        'size': 2,
                        'color': "green",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = subpoints[:,0],
                     y = subpoints[:,1],
                     z = subpoints[:,2],
                    mode='markers', type='scatter3d',
                    marker={
                        'size': 2,
                        'color': "blue",
                        'colorscale':'rainbow',
}),
        go.Scatter3d(x = bb[:,0],
                     y = bb[:,1],
                     z = bb[:,2],
                    mode='lines', type='scatter3d',
                    line={
                        'width': 10,
                        'color': "red",
                        'colorscale':'rainbow'
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


# In[41]:


root = f"Dataset 2022{record_number}/"
pts_blick_rec1 = []
distance = []
pts_velo_rec1 = []
pts_radar_rec1 = []
for i in range(240):
    pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    label = readLabels(f"{root}/groundtruth/{i:06d}.csv")
    subpoints = get_sub_points_of_object(label, pc_blick[:,0:3])
    pts_blick_rec1.append(len(subpoints))
    subpoints = get_sub_points_of_object(label, pc_velo[:,0:3])
    pts_velo_rec1.append(len(subpoints))
    subpoints = get_sub_points_of_object(label, pc_radar[:,0:3])
    pts_radar_rec1.append(len(subpoints))
    dist=distance_of_center_points(label)
    distance.append(dist)
    
print(pts_velo_rec1)
#print(distance)
    
    #pts_blick_rec1 += [get_sub_points_of_object(label, pc_blick[:,0:3])[0].sum()]
    #pts_velo_rec1 += [get_sub_points_of_object(label, pc_velo[:,0:3])[0].sum()]
    #pts_radar_rec1 += [get_sub_points_of_object(label, pc_radar[:,0:3])[0].sum()]


# In[42]:


plt.scatter(distance,pts_blick_rec1)
plt.scatter(distance,pts_velo_rec1)
plt.scatter(distance,pts_radar_rec1)
plt.legend(["Blick", "Velodyne", "Radar"])


# In[43]:


root = f"Dataset 2022{record_number}/"
pts_blick_rec2 = []
pts_velo_rec2 = []
pts_radar_rec2 = []
distance1 = []
for i in range(100):
    pc_blick = readPC(f"{root}/blickfeld/{i:06d}.csv")
    pc_velo = readPC(f"{root}/velodyne/{i:06d}.csv")
    pc_radar = readPC(f"{root}/radar/{i:06d}.csv")
    label = readLabels(f"{root}/groundtruth/{i:06d}.csv")
    subpoints = get_sub_points_of_object(label, pc_blick[:,0:3])
    pts_blick_rec2.append(len(subpoints))
    subpoints = get_sub_points_of_object(label, pc_velo[:,0:3])
    pts_velo_rec2.append(len(subpoints))
    subpoints = get_sub_points_of_object(label, pc_radar[:,0:3])
    pts_radar_rec2.append(len(subpoints))
    dist=distance_of_center_points(label)
    distance1.append(dist)
    
#print(pts_blick_rec1)
print(distance1)
 
# In[44]:


plt.scatter(distance1,pts_blick_rec2)
plt.scatter(distance1,pts_velo_rec2)
plt.scatter(distance1,pts_radar_rec2)
plt.legend(["Blick", "Velodyne", "Radar"])





