#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:22:19 2018

@author: cls0208
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:59:40 2018

@author: cls0208
"""
'''
Dependancies:
    
    First, we need the cloud material. The current version is Cloudmatv0.1. This can be saved onto a 
    "Mat Saver" object so that the material does not dissapear when accidentally deleted.
    
    Next, we neeed a floor. Simply a 16x16BU plane with some roughness

'''
import sys
    
    
'''
Now we can pull netCDF4 and proper verions of numpy to utlize in the code.
'''

import netCDF4 as nc
import numpy as np

#from fractalgen import alpha_shape3d, cloudfieldplot
import bpy
import bmesh
import matplotlib.pyplot as plt
import os


def trackpull(dset):
    sets=np.array([])
    for file in os.listdir('%s'%(dset)): 
        if file.endswith('track2.nc') or file.endswith('track.nc'): #pulling all track files from the folder
            fname='%s/%s'%(dset,file)
            f=nc.Dataset(fname,'r')
            if 'nr' in (f.variables) and (len(f.variables['nr'][:]))>1: #checking to see if there are clouds in the file
                sets=np.append(sets,file) #putting file names into a list
            f.close()
    sets=np.sort(sets)
    return sets

def cloudsort(netcdffile):
    f = netcdffile
    # The following algorithm is borrowed from Cloud-Base 1
    # This code finds all of the clouds and sorts them into a large array (z,y,x,ID) z,y,x are coordinates and ID is the clouds number.
    nrcloud= f.variables['nrcloud'][:]
    #array of the size of all points that contain clouds
    totcloudpoints = np.nonzero(nrcloud)[0].size   

    idx = np.zeros((totcloudpoints,4)) #idx is a 2D array of sizes totcloudpoints and 4, or the dimensions of nrcloud: z,y,x,nrcloud value (1 to nr.size)
    tmp =  np.nonzero(nrcloud) #indices of nrcloud that are nonzero   
    idx[:,0] = tmp[0]   #z coordinates of nrcloud
    idx[:,1] = tmp[1]   #y coordinates of nrcloud
    idx[:,2] = tmp[2]   #x coordinates of nrcloud
    
        
    for j in range(int(totcloudpoints)):

        idx[j,3] = nrcloud[int(idx[j,0]),int(idx[j,1]),int(idx[j,2])]  #gives the fourth index, the nrcloud value, from idx: 1 to nr.size
    
    idx2 = idx[idx[:, 3].argsort()] #makes this idx2 array the sorted version of idx 
    return idx2

directory = '/data'
dset = '/wrf_microhh/20160819/'

dset = directory+dset
sets = trackpull(dset)

fname = dset+sets[3]
#qlname = dset+'fielddump.ql.00.04.nc'
f=nc.Dataset(fname,'r')
#ql = nc.Dataset(dset+'fielddump.ql.00.04.track.nc')
varname = "nrcloud" 
idx2 = cloudsort(f)     
num = int(np.max(idx2[:, 3]))
rr=idx2

nrcloud = f.variables['nrcloud'][:]
merged_nrcloud = nrcloud.sum(axis = 1).sum(axis = 1)
CBH = 25 * np.where(merged_nrcloud == merged_nrcloud[merged_nrcloud>0][0])[0]
CBH =  int(CBH)

'''
-----
'''
bpy.ops.mesh.primitive_cube_add(radius=8*0.999/1024);
orig_cube = bpy.context.active_object;
#mat = bpy.data.materials.get('CloudMatv0.1')
mat = bpy.data.materials.get('CloudMask')

#print(mat)

orig_cube.data.materials.append(mat)
orig_cube.name = 'Cloud Base'
'''
-----
'''
bpy.ops.mesh.primitive_cube_add(radius=8*0.999/1024);
orig_cube_sides = bpy.context.active_object;
#mat = bpy.data.materials.get('CloudMatv0.1')
mat = bpy.data.materials.get('BaseMask')

#print(mat)

orig_cube_sides.data.materials.append(mat)
orig_cube_sides.name = 'Cloud Side'
'''
------
desired camera location
'''
ycoordstoloop = np.arange(4,12,2) - 8
xcoordstoloop = np.arange(4,12,2) - 8

for l in xcoordstoloop:
    for m in ycoordstoloop:
        
        k = 0
        meancloudbase = 0
        bpy.ops.mesh.primitive_plane_add();
        o = bpy.context.active_object;
        me = o.data;
        bm = bmesh.new();
        o.name = 'Cloud Sides'
        
        bpy.ops.mesh.primitive_plane_add();
        osides = bpy.context.active_object;
        mesides = osides.data;
        bmsides = bmesh.new();
        osides.name = 'Cloud Bases'
        
        print('-----Moving Camera-----')
        camera_loc = np.array([l,m,0.001])
        camera = bpy.data.objects['TSI']
        camera.location.x = camera_loc[0]
        camera.location.y = camera_loc[1]
        camera.location.z = camera_loc[2]
        plt.figure()
        print('-----Placing Cloud Particles-----')
        for i in range(num): #number of clouds to loop over set to 'num' if you want to loop over all
            #print(i)
            indcloud = rr[rr[:,3]==i+1]
            xs = indcloud[:,2]*16/1024 -8 
            ys = indcloud[:,1]*16/1024 -8
            zs = indcloud[:,0]*16/1024
#            zsmin = np.min(zs)
            zslim = np.min(zs) + 100/1600
            
            meanx = np.mean(xs)
            meany = np.mean(ys)
            meanz = np.mean(zs)
            dx = meanx - camera_loc[0]
            dy = meany - camera_loc[1]
            dz = meanz - camera_loc[2]
            r = np.sqrt(dx**2+dy**2)
            theta = np.arctan(r/dz) * 180/np.pi
            if theta < 80:
                k += 1
                meancloudbase = (meancloudbase*(k-1)+min(zs))/k
                #rendercount += 1
                for i in range(len(xs)):
                    
                    #print(meancloudbase)
                    x = xs[i]
                    y = ys[i]
                    z = zs[i]
                    
                    
                    if zs[i] > zslim:
                        bm.verts.new().co= [x, y, z]
                        a = 'balderdash'
                    else:
                        bmsides.verts.new().co = [x, y, z]
                        
                    
                    
                    
                    
                bm.to_mesh(me);
                o.dupli_type = 'VERTS';
                orig_cube.parent = o;  
        
        
                bmsides.to_mesh(mesides);
                osides.dupli_type = 'VERTS';
                orig_cube_sides.parent = osides;  
 
        #text = open('/users/PFS0220/cls0208/Renders/New/Image Data(loc =('+str(l)+', '+str(m)+').txt','w')
        #text.write(['Mean cloud base: '+str(meancloudbase*1024/16*25)+' m','xloc = '+str((l+8)*1024/16*25)+' m','yloc = '+str((m+8)*1024/16*25)+' m'])
        #text.close()
        print('-----Rendering-----')        
        bpy.data.scenes['Scene'].render.filepath = '/home/nick/Desktop/Renders/TSI_R (loc =('+str(l)+', '+str(m)+')).png'
        bpy.ops.render.render( write_still=True )
        
        
        print('-----Cleaning Up-----')
        bpy.ops.object.select_all(action = 'DESELECT')
        bpy.data.objects['Cloud Bases'].select = True
        bpy.ops.object.delete()
        
        bpy.data.objects['Cloud Sides'].select = True
        bpy.ops.object.delete()