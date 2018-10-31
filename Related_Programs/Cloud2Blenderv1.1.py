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
import sys

#import matplotlib.pyplot as plt
'''
User Defined Options
'''
import argparse
argv = sys.argv
if "--" not in argv:
    argv = [] # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:] # get all args after "--"
    print(argv)
usage_text = "Run blender in background mode with this script: blender --background --python " + __file__ + " -- [options]"
parser = argparse.ArgumentParser(description=usage_text)

parser.add_argument("dataset",
                    help="Path to the data set")
parser.add_argument("-d", "--device",
                    help="GPU or CPU rendering", default = "GPU")
parser.add_argument("-t", "--type",
                    help="pretty or masked images", default = "masked")
args = parser.parse_args(argv) # In this example we wont use the args


print(args.dataset)
print(args.device)
print(args.type)
dset = args.dataset # setting the directory containing ql.nc
render_landing = dset
#render_type = "masked"
#render_type = "pretty"
blender_file_masked = "/data/scripts/LES_TSI/Cloudmapper_labels.blend"
blender_file_pretty = "/data/scripts/LES_TSI/Cloudmapper_pretty.blend"
width = 6400.  # The extent left and right that needs to be processed
ncameras = 5 # The number of camera lines in the x direction
dy_cam  = 150. # The spatial step that the camera makes each rendered frame (i.e., the time step of the TSI times the mean wind)

'''
Source
 |
 V
'''

if (args.type == "masked"):
  bpy.ops.wm.open_mainfile(filepath=blender_file_masked)
else:
  bpy.ops.wm.open_mainfile(filepath=blender_file_pretty)


#timesim = timeindex * 3600 # setting simulation time
f = nc.Dataset(dset+'ql.nc') # loading in the file containing liquid water
time = f.variables['time'][:]
x = f.variables['x'][:]
y = f.variables['y'][:]
z = f.variables['z'][:]

dx = x[1]-x[0]
dy = y[1]-y[0]
dz = z[1]-z[0]

Lx = dx*x.size
Ly = dy*y.size
Lz = dz*z.size

width_BU = width * 16/Lx
nwidth = int(width/dx)
icameras = np.linspace(nwidth, x.size - nwidth, num = ncameras, dtype = int)


'''
Set the camera, rendering, vertex definitions
'''

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = args.device
scene.render.image_settings.file_format = 'AVI_RAW'
scene.render.resolution_x = 480 # pixels
scene.render.resolution_y = 640

bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)

bpy.ops.object.camera_add(location = (0,0,0), rotation = (np.pi, 0, 0,))
bpy.data.objects['Camera'].name = 'TSI'
camera = bpy.context.object.data
camera.type  = 'PANO'
camera.cycles.panorama_type = 'FISHEYE_EQUISOLID'
camera.cycles.fisheye_fov = 2.79 #2.79rad = 160degree
camera.cycles.fisheye_lens = 10.5
camera.sensor_width = 36

scene.cycles.transparent_max_bounces = 80
scene.cycles.transparent_min_bounces = 20
scene.cycles.max_bounces = 80
scene.cycles.transmission_bounces = 80
scene.cycles.volume_bounces = 80

scene.cycles.volume_step_size = 2 * 16/y.size
scene.cycles.volume_max_steps = 256

scene.cycles.sample_clamp_direct = 10
scene.cycles.sample_clamp_indirect = 3
scene.render.tile_x = scene.render.resolution_x 
scene.render.tile_x = scene.render.resolution_y 



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
bpy.ops.mesh.primitive_cube_add(radius=16*0.499999/y.size);
orig_cube = bpy.context.active_object;
mat = bpy.data.materials.get('BaseMask')
orig_cube.data.materials.append(mat)
orig_cube.name = 'Cloud Base'

bpy.ops.mesh.primitive_cube_add(radius=16*0.49999/y.size);
orig_cube_sides = bpy.context.active_object;
mat = bpy.data.materials.get('CloudMask')
orig_cube_sides.data.materials.append(mat)
orig_cube_sides.name = 'Cloud Side'

bpy.data.scenes["Scene"].cycles.device = args.device



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

'''
scene.world.use_nodes = True

#select world node tree
wd = scene.world
nt = bpy.data.worlds[wd.name].node_tree

#create new Sky texture node
skyNode = nt.nodes.new(type="ShaderNodeTexSky")
skyNode.turbidity  = 3.
#create new Sky texture node
BGNode = nt.nodes.new(type="ShaderNodeTexSky")
BGNode.  = 3.

#find location of Background node and position Grad node to the left
backNode = nt.nodes['Background']
skyNode.location.x = backNode.location.x-300
skyNode.location.y = backNode.location.y

#Connect color out of Grad node to Color in of Background node
SkyColOut = skyNode.outputs['Color']

backColIn = backNode.inputs['Color']
nt.links.new(SkyColOut, backColIn)
'''

for t in range(time.size):
  print("Time  = ", time[t])
  ql = f.variables['ql'][t,:,:,:] # loading the liquid water into memory
  if(np.count_nonzero(ql) == 0):
    continue


  #qlsliced = ql[timeindex,:,:,:] # trimming down ql to only include the time needed
  ql = np.round((ql/np.min(ql[ql>0]))**(1./4.)) # weighting the data
  #f.close()
  #del ql
  print('-----Placing Cloud Particles-----')

  for icam in (icameras):
    print ("Camera at x =", icam*dx)
    icam_BU = (icam / x.size - 0.5) * 16
    
    qlmask_bot = np.zeros_like(ql)
    qlmask_side = (ql > 0)

    qlmask_side[:,:,:icam - nwidth] = 0
    qlmask_side[:,:,icam + nwidth:] = 0
    
    for i in range (ql.shape[2]):
        for j in range (ql.shape[1]):
            k = np.nonzero(qlmask_side[:,j,i])[0]
            for n in range (min(1,k.size)):
              qlmask_bot[k[n],j,i]  = 1
              qlmask_side[k[n],j,i] = 0

            
    for k,j,i in  np.array(qlmask_bot[:,:,:].nonzero()).T:
#        iBU = (i / x.size - 0.5) * 16
        iBU = ((i-icam) / x.size) * 16 + 2 # scaling to bring things into blender space from simulation space
#        print(iBU, i, icam, x.size)
        jBU = (j / y.size - 0.5) * 16
        kBU = k * dz / Lx * 16

        place_verts(bm_points,np.array([kBU, jBU, iBU]), int(ql[k, j, i]))

        bm.verts.new().co= [iBU, jBU, kBU]
        
    for k,j,i in  np.array(qlmask_side[:,:,:].nonzero()).T:
#        iBU = ((i-icam) / x.size - 0.5) * 16 + 2 # scaling to bring things into blender space from simulation space
        iBU = ((i-icam) / x.size) * 16 + 2 # scaling to bring things into blender space from simulation space
        jBU = (j / y.size - 0.5) * 16
        kBU = k * dz / Lx * 16
        place_verts(bm_points,np.array([kBU, jBU, iBU]), int(ql[k, j, i]))

        bmsides.verts.new().co= [iBU, jBU, kBU]

    del qlmask_bot, qlmask_side
                
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

    ycoords = np.arange(-8+width_BU,8-width_BU,dy_cam * 16/Ly) # every 150m (30s x 5m/s) step size


    scene = bpy.context.scene
    frame = 1

    for ob in bpy.data.objects:
        ob.select = False


    #MASKED RENDER

    if(args.type == "masked"):
      bpy.data.objects['Cloud Volume'].hide_render = True
      bpy.data.objects['Cloud Points'].hide_render = True
      bpy.data.objects['Cloud Sides'].hide_render = False
      bpy.data.objects['Cloud Bases'].hide_render = False
          
      bpy.data.objects['TSI'].select = True # putting the TSI as the acitve object so we only animate it.

      frame = 0
      for m in ycoords:
          camera_loc = np.array([icam_BU,m,0.001])
          print(camera_loc)
          camera = bpy.data.objects['TSI']
          scene.frame_set(frame) # updating the fram
          camera.location.x = camera_loc[0] # move the camera to appropriate x,y,z locs
          camera.location.y = camera_loc[1]
          camera.location.z = camera_loc[2]
          
          #bpy.ops.anim.keyframe_insert(type = 'Location') # Creating keyframe
          
          frame += 1
          
      
          
      bpy.data.scenes['Scene'].render.filepath = render_landing+str(time[t])+'/TSI_R_MASKED (along x = '+'{:.0f}'.format(icam * dx)+').avi' # setting save location
      bpy.ops.render.render( write_still=True, animation = True ) # initializing rendering

    #PRETTY RENDER
    
    if(args.type == "pretty"):
      bpy.data.objects['Cloud Volume'].hide_render = False
      bpy.data.objects['Cloud Points'].hide_render = False
      bpy.data.objects['Cloud Sides'].hide_render = True
      bpy.data.objects['Cloud Bases'].hide_render = True

      for ob in bpy.data.objects:
          ob.select = False
          
      bpy.data.objects['TSI'].select = True # putting the TSI as the acitve object so we only animate it.

      frame = 0
      for m in ycoords:
          camera_loc = np.array([icam_BU,m,0.001])
          print(camera_loc)
          camera = bpy.data.objects['TSI']
          scene.frame_set(frame) # updating the fram
          camera.location.x = camera_loc[0] # move the camera to appropriate x,y,z locs
          camera.location.y = camera_loc[1]
          camera.location.z = camera_loc[2]
          
          #bpy.ops.anim.keyframe_insert(type = 'Location') # Creating keyframe
          
          frame += 1
          
      
          
      bpy.data.scenes['Scene'].render.filepath = render_landing+str(time[t])+'/TSI_R (along x = '+'{:.0f}'.format(icam * dx)+').avi' # setting save location
      bpy.ops.render.render( write_still=True, animation = True ) # initializing rendering
