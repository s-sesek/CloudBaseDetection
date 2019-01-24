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
print('------ NEW RENDERING SESSION ------')
#given ql let's make the new system to render pretty images based on vertex density
''' 
These next couple of dozen lines represent the 'plotting node'
'''
print('----- Placing Cloud Particles -----')
import bpy
import bmesh
import numpy as np
import netCDF4 as nc
import os

dset = '/data/wrf_microhh/20160611/' # setting the directory containing ql.nc

timeindex = 8 # setting index time
timesim = timeindex * 3600 # setting simulation time





f = nc.Dataset(dset+'ql.nc') # loading in the file containing liquid water
ql = f.variables['ql'][:] # loading the liquid water into memory

qlsliced = ql[timeindex,:,:,:] # trimming down ql to only include the time needed
qlnew = np.round((qlsliced/np.min(qlsliced[qlsliced>0]))**(1./4.)) # weighting the data
#qlnew = np.round( qlsliced/ np.max(qlsliced) * 300)
f.close()
del ql

def place_verts(bm, coords, number):
    '''
    This function is used to plot the vertices into blender with the weights 
    as assigned by number.
    '''
    for i in range(number):
        bm.verts.new().co= [coords[2], coords[1], coords[0]]  
        
zbounds = qlnew.shape[0] # assigning the range for the plotting loop to go over


bpy.ops.mesh.primitive_plane_add();  # building the mesh to assign verts to
o = bpy.context.active_object;
me = o.data;
bm = bmesh.new();
o.name = 'Cloud Points' # renaming it to something useful

for k in range(zbounds): 
    print(np.round(k/zbounds* 100, 1), '%')
    mask = np.array(np.where(qlnew[k,:,:] > 0)).T # finding where liquid water exists
    
    for j,i in mask:
        x = i * 0.015625-8 # scaling to bring things into blender space from simulation space
        y = j * 0.015625-8
        z = k * 0.015625
        place_verts(bm,np.array([z, y, x]), int(qlnew[k, j, i]))
        
bm.to_mesh(me) # assigning the vertices to the mesh defined above
o.dupli_type = 'VERTS' # the type of object at each point
print('done')
'''
----------
'''


'''
Assigning materials and volumetric densities
'''
print('----- Assigning materials and volumetric Densities -----')
mat2 = bpy.data.materials['Cloud Volume'] # pulling the material variable into memory
mat2.use_nodes = True # nodes are how the materials are changed
mat2.node_tree.nodes[3].object = bpy.data.objects['Cloud Points'] # assigning the 'Object' parameter to the 'Cloud Points' we jsut made

print('done')


'''
----------
'''



'''
This node is for the camera positioning and rendering 
'''

print('----- Rendering -----')

ycoords = np.arange(0,25600,150)/ 25 /1024 * 16 - 8 # every 150m (30s x 5m/s) step size
xcoords = np.arange(4,12,2) - 8
scene = bpy.context.scene
frame = 1

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
        
        bpy.ops.anim.keyframe_insert(type = 'Location') # Creating keyframe

        
        print(frame, camera_loc)
        
        frame += 1
        
    
        
    bpy.data.scenes['Scene'].render.filepath = '/home/nick/Desktop/Renders/wrf_microhh/20160611_old/'+str(timesim)+'/TSI_R (along x = '+str((l + 8) * 1600)+').avi' # setting save location
    bpy.ops.render.render( write_still=True, animation = True ) # initializing rendering
