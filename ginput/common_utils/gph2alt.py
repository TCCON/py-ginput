import numpy as np


def auto_broadcast(arr, target):
    arr = np.asarray(arr)
    target = np.asarray(target)

    # find which axis of target matches arr length
    axis = None
    for ax, size in enumerate(target.shape):
        if size == arr.size:
            axis = ax
            break

    if axis is None:
        raise ValueError("No matching dimension found")

    # build shape
    shape = [1] * target.ndim
    shape[axis] = arr.size
    return arr.reshape(shape)



#lat               I   degrees
#surfs    O   
def surface_gravity(lat):
    
    lat = np.asarray(lat, dtype=np.float32)
    g1 = 9.80616
    g2 = 2.6373e-3
    g3 = 5.9e-6

    cos2lat = np.cos(np.deg2rad(2.0 * lat))

    surfs = g1 * (1.0 - g2 * cos2lat) + g3 * (cos2lat ** 2)
    return surfs

#lat    I   degrees
#surfs  I   output of surface_gravity
def calc_eff_rad(surfs, lat):
    lat = np.asarray(lat, dtype=np.float32)
    e1 = 3.085462e-6
    e2 = 2.27e-9
    e3 = 2e-12

    cos2lat = np.cos(np.deg2rad(2.0 * lat))
    cos4lat = np.cos(np.deg2rad(4.0 * lat))
   
    result = 2.0 * surfs / (e1 + e2 * cos2lat - e3 * cos4lat) / 1000.0
    return result



def convert_gph_to_alt(gph, lat):

    gph = np.asarray(gph, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float32)

    if np.shape(gph) != np.shape(lat):    
        raise ValueError(f"Shape mismatch between gph and lat")
        
    surfs = surface_gravity(lat)
    r = calc_eff_rad(surfs, lat)

    
    g0 = 9.80665

    
    alt = r / (1.0 - (gph * g0) / (r * surfs)) - r
    
    return alt


