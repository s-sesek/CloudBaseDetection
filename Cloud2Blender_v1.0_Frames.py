import netCDF4 as nc
import numpy as np
import bpy
import bmesh

# - Setting the directory containing ql.nc
qlncDir = './blender/'

# - Path to masked blender file
blender_file_masked = "./blender/Cloudmapper_labels.blend"

# - Path to pretty blender file
blender_file_pretty = "./blender/Cloudmapper_pretty.blend"

# - Type of render, "masked", or "pretty"
render_type = "masked"

# - Output directory
render_landing = qlncDir

# The extent left and right that needs to be processed
width = 6400.

# The number of camera lines in the x direction
numberOfCameras = 5

# The spatial step that the camera makes each rendered frame
# (i.e., the time step of the TSI times the mean wind)
dy_cam = 150.

# - Check to see if we are doing a masked or pretty render to open the corresponding blender file
if render_type == "masked":
    bpy.ops.wm.open_mainfile(filepath=blender_file_masked)
else:
    bpy.ops.wm.open_mainfile(filepath=blender_file_pretty)


# - Loading in the file containing liquid water
f = nc.Dataset(qlncDir + 'ql.nc')

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
cameras = np.linspace(nwidth, x.size - nwidth, num=numberOfCameras, endpoint=False, dtype=int)


def place_verts(bm, coords, number):
    '''
    This function is used to plot the vertices into blender with the weights 
    as assigned by number.
    '''
    for i in range(number):
        bm.verts.new().co = [coords[2], coords[1], coords[0]]


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


for t in range(10, time.size):
    print("Time  = ", time[t])

    # - Loading the liquid water into memory
    ql = f.variables['ql'][t, :, :, :]

    if ql.max() == 0:
        continue

    # - Weighting the data
    ql = np.round((ql/np.min(ql[ql > 0]))**(1./4.))

    print('-----Placing Cloud Particles-----')
    qlmask_bot = np.zeros_like(ql)
    qlmask_side = (ql > 0)

    for i in range(ql.shape[2]):
        for j in range(ql.shape[1]):
            k = np.nonzero(qlmask_side[:, j, i])[0]
            for n in range(min(1, k.size)):
                qlmask_bot[k[n], j, i] = 1
                qlmask_side[k[n], j, i] = 0

    print("MAX Bottom Coordinates", np.array(qlmask_bot[:, :, :].nonzero()).max(axis=1))
    for k, j, i in np.array(qlmask_bot[:, :, :].nonzero()).T:
        # - Scaling to bring things into blender space from simulation space
        iBU = (i / x.size - 0.5) * 16
        jBU = (j / y.size - 0.5) * 16
        kBU = k * dz / Lx * 16
        place_verts(bm_points, np.array([kBU, jBU, iBU]), int(ql[k, j, i]))
        bm.verts.new().co = [iBU, jBU, kBU]
      
    print("MAX Side Coordinates", np.array(qlmask_side[:, :, :].nonzero()).max(axis=1))
    for k, j, i in np.array(qlmask_side[:, :, :].nonzero()).T:
        # - Scaling to bring things into blender space from simulation space
        iBU = (i / x.size - 0.5) * 16
        jBU = (j / y.size - 0.5) * 16
        kBU = k * dz / Lx * 16
        place_verts(bm_points, np.array([kBU, jBU, iBU]), int(ql[k, j, i]))
        bmsides.verts.new().co = [iBU, jBU, kBU]

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
    # pulling the material variable into memory
    mat2 = bpy.data.materials['Cloud Volume']

    # nodes are how the materials are changed
    mat2.use_nodes = True

    # assigning the 'Object' parameter to the 'Cloud Points' we jsut made
    mat2.node_tree.nodes[13].object = bpy.data.objects['Cloud Points']

    print('done')

    print('----- Rendering -----')
    # every 150m (30s x 5m/s) step size
    ycoords = np.arange(-8+width_BU,8-width_BU,dy_cam * 16/Ly)

    # - Set the scene
    scene = bpy.context.scene

    for ob in bpy.data.objects:
        ob.select = False

    # Set output to PNG for frames
    scene.render.image_settings.file_format = 'PNG'

    # - Putting the TSI as the active object so we only animate it.
    bpy.data.objects['TSI'].select = True

    for cam in cameras:
        print("Camera at x =", cam * dx)

        icam_BU = (cam / x.size - 0.5) * 16

        # - Set which objects to show/hide based on render type
        if render_type == "masked":
            bpy.data.objects['Cloud Volume'].hide_render = True
            bpy.data.objects['Cloud Points'].hide_render = True
            bpy.data.objects['Cloud Sides'].hide_render = False
            bpy.data.objects['Cloud Bases'].hide_render = False
        else:
            bpy.data.objects['Cloud Volume'].hide_render = False
            bpy.data.objects['Cloud Points'].hide_render = False
            bpy.data.objects['Cloud Sides'].hide_render = True
            bpy.data.objects['Cloud Bases'].hide_render = True

        # Start at frame 0
        frame = 0

        for m in ycoords:
            camera_loc = np.array([icam_BU, m, 0.001])
            print(camera_loc)
            camera = bpy.data.objects['TSI']

            # - Set the frame
            scene.frame_set(frame)

            # - Move the camera to appropriate x,y,z locations
            camera.location.x = camera_loc[0]
            camera.location.y = camera_loc[1]
            camera.location.z = camera_loc[2]

            # Set save location for the blender output
            bpy.data.scenes['Scene'].render.filepath = render_landing + str(time[t]) + '/' + render_type + '(along x = ' + '{:.0f}'.format(cam * dx) + ') - FRAME' + str(frame) + '.png'

            # - Render the frame
            bpy.ops.render.render(write_still=True)

            # - Advance the frame
            frame += 1
