# Copyright 2018 Edward Allums, Gopal Godhani, Dan Mayich, and Sarah Sesek
# Copyright 2018 Nick Barron.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Version 1.0 - June 5, 2018

@author: Nick Barron
"""


'''
Initializing objects and retrieving data
'''
import netCDF4 as nc
import numpy as np
import bpy
import bmesh
import os
#import matplotlib.pyplot as plt
'''
User Defined Options
'''
dset = '/data/lasso/sims/20160611/' # setting the directory containing ql.nc
timeindex =  8 # setting index time
render_landing = '/data/lasso/sims/20160611/' #file output locatioon

'''
Source
 |
 V
'''



timesim = timeindex * 3600 # setting simulation time
f = nc.Dataset(dset+'ql.nc') # loading in the file containing liquid water
ql = f.variables['ql'][:] # loading the liquid water into memory
z = f.variables['z'][:]

qlsliced = ql[timeindex,:,:,:] # trimming down ql to only include the time needed
qlnew = np.round((qlsliced/np.min(qlsliced[qlsliced>0]))**(1./4.)) # weighting the data
CBH_idx = np.min(np.where(qlsliced.mean(axis = 1).mean(axis = 1)>0)[0]) 
print(np.where(qlsliced.mean(axis = 1).mean(axis = 1)>0))
CBH = z[CBH_idx]
f.close()
del ql

def place_verts(bm, coords, number):
    '''
    This function is used to plot the vertices into blender with the weights 
    as assigned by number.
    '''
    for i in range(number):
        bm.verts.new().co= [coords[2], coords[1], coords[0]]  

'''
-----
'''
bpy.ops.mesh.primitive_cube_add(radius=8*0.999/1024);
orig_cube = bpy.context.active_object;
mat = bpy.data.materials.get('BaseMask')
orig_cube.data.materials.append(mat)
orig_cube.name = 'Cloud Base'

bpy.ops.mesh.primitive_cube_add(radius=8*0.999/1024);
orig_cube_sides = bpy.context.active_object;
mat = bpy.data.materials.get('CloudMask')
orig_cube_sides.data.materials.append(mat)
orig_cube_sides.name = 'Cloud Side'

#-------
bpy.ops.mesh.primitive_plane_add();
o = bpy.context.active_object;
me = o.data;
bm = bmesh.new();
o.name = 'Cloud Bases'
#-------
bpy.ops.mesh.primitive_plane_add();
osides = bpy.context.active_object;
mesides = osides.data;
bmsides = bmesh.new();
osides.name = 'Cloud Sides'
#--------
bpy.ops.mesh.primitive_plane_add();  # building the mesh to assign verts to
o_points = bpy.context.active_object;
me_points = o_points.data;
bm_points = bmesh.new();
o_points.name = 'Cloud Points' # renaming it to something useful

k = 0
meancloudbase = 0
print('-----Placing Cloud Particles-----')
zslim = (CBH + 360) / 1600
print(zslim)
zbounds = np.shape(qlsliced)[0]
print(zbounds)
for k in range(zbounds): 
    
    print(np.round(k/zbounds* 100, 1), '%')
    mask = np.array(np.where(qlsliced[k,:,:] > 0)).T # finding where liquid water exists
 
    for j,i in mask:
        
        x = i * 0.015625-8 # scaling to bring things into blender space from simulation space
        y = j * 0.015625-8
        z = k * 40/1600
        place_verts(bm_points,np.array([z, y, x]), int(qlnew[k, j, i]))

        if z <= zslim:  
            #print('a')  
            bm.verts.new().co= [x, y, z]
        if z > zslim:
            #print('yay')
            bmsides.verts.new().co= [x, y, z]
            
bm.to_mesh(me);
o.dupli_type = 'VERTS';
orig_cube.parent = o;  


bmsides.to_mesh(mesides);
osides.dupli_type = 'VERTS';
orig_cube_sides.parent = osides;  

bm_points.to_mesh(me_points) # assigning the vertices to the mesh defined above
o_points.dupli_type = 'VERTS' # the type of object at each point



'''
Assigning materials and densities
'''
print('----- Assigning materials and volumetric Densities -----')
mat2 = bpy.data.materials['Cloud Volume'] # pulling the material variable into memory
mat2.use_nodes = True # nodes are how the materials are changed
mat2.node_tree.nodes[13].object = bpy.data.objects['Cloud Points'] # assigning the 'Object' parameter to the 'Cloud Points' we jsut made

print('done')


'''
----------
'''
'''
This node is for the camera positioning and rendering 
'''

print('----- Rendering -----')

ycoords = np.arange(0,25600,150)/ 25 /1024 * 16 - 8 # every 150m (30s x 5m/s) step size


ycoords = ycoords[30:-30]


xcoords = np.arange(4,12,2) - 8
scene = bpy.context.scene
frame = 1

for ob in bpy.data.objects:
    ob.select = False

'''
#MASKED RENDER
'''

bpy.data.objects['Cloud Volume'].hide_render = True
bpy.data.objects['Cloud Points'].hide_render = True
bpy.data.objects['Cloud Sides'].hide_render = False
bpy.data.objects['Cloud Bases'].hide_render = False

    
bpy.data.objects['TSI'].select = True # putting the TSI as the acitve object so we only animate it.
for l in xcoords:
    frame = 0
    for m in ycoords:
        camera_loc = np.array([l,m,0.001])
        camera = bpy.data.objects['TSI']
        scene.frame_set(frame) # updating the fram
        camera.location.x = camera_loc[0] # move the camera to appropriate x,y,z locs
        camera.location.y = camera_loc[1]
        camera.location.z = camera_loc[2]
        
        #bpy.ops.anim.keyframe_insert(type = 'Location') # Creating keyframe
         #commented out, created unknown error
        
        print(frame, camera_loc)
        
        frame += 1
        
    
        
    bpy.data.scenes['Scene'].render.filepath = render_landing+str(timesim)+'/TSI_R_MASKED (along x = '+str((l + 8) * 1600)+').avi' # setting save location
    bpy.ops.render.render( write_still=True, animation = True ) # initializing rendering
  
'''
#PRETTY RENDER
'''   
'''
bpy.data.objects['Cloud Volume'].hide_render = False
bpy.data.objects['Cloud Points'].hide_render = False
bpy.data.objects['Cloud Sides'].hide_render = True
bpy.data.objects['Cloud Bases'].hide_render = True
  
for ob in bpy.data.objects:
    ob.select = False
    
bpy.data.objects['TSI'].select = True # putting the TSI as the acitve object so we only animate it.

for l in xcoords:
    frame = 0
    for m in ycoords:
        camera_loc = np.array([l,m,0.001])
        camera = bpy.data.objects['TSI']
        scene.frame_set(frame) # updating the fram
        camera.location.x = camera_loc[0] # move the camera to appropriate x,y,z locs
        camera.location.y = camera_loc[1]
        camera.location.z = camera_loc[2]
        
        #bpy.ops.anim.keyframe_insert(type = 'Location') # Creating keyframe
         #commented out, created unknown error
        
        print(frame, camera_loc)
        
        frame += 1
        
    
        
    bpy.data.scenes['Scene'].render.filepath = render_landing+str(timesim)+'/TSI_R (along x = '+str((l + 8) * 1600)+').avi' # setting save location
    bpy.ops.render.render( write_still=True, animation = True ) # initializing rendering
'''
