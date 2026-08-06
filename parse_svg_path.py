# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


dx = 0.10
dy = 0.12
x_min = 0.12
z_min = 0.30



def parse_svg_for_paths(svg_file):
    """
    Parses the line paths in an SVG file (marked "d") 
    into a set of absolute spatial coordinates. 
    """

    # Find lines with 'd="' at the beginning: 
    d_lines = []
    ignore = 0    # control parameter to ignore "<clipPath>" sections
    with open(svg_file,'r') as svgfile:
        for line in svgfile:
            l = line.strip()
            if l[0:3]=='d="' and not ignore:
                d_lines.append(line.strip().strip(' <d="/>'))
            elif l[0:9] == '<clipPath':
                ignore = 1
            elif l[0:11] == '</clipPath>':
                ignore = 0
            
    # parse such lines to separate out anything delimited by space
    svg_coords = np.ndarray((0,3)).astype(float)
    d_line_origin = np.array([[0.,0.,0.]])
    current_point = np.array([[0.,0.,0.]])
    #first_absolute = 1 
    for d_line in d_lines:
        chunks = d_line.split(' ')  # Split the string at spaces

        # Step through the chunks and parse out coordinates
        ii = 0
        while ii < len(chunks) : 
            chunk = chunks[ii]

            if chunk[0].isalpha() :  # If chunk is alphabetic, it's a setting telling us what's coming. 
                
                # Case of mode command determines absolute (upper) or relative (lower)
                if chunk.isupper():
                    absolute = 1
                else : 
                    absolute = 0
                
                # use the lowercase version to settle what mode we are in. 
                mode = chunk.lower()
                # Set the number of chunks to increment
                if mode == "m":
                    incr = 1
                    # first M or m is always absolute, but this is a separate setting from Mode. 
                    first_absolute = 1
                elif mode == "l":
                    incr = 1
                elif mode == "h":  # horizontal segment
                    incr = 1
                elif mode == "v":  # vertical segment
                    incr = 1
                elif mode == "c":  # skip the control points
                    incr = 3
                elif mode == "s":  # skip the control point
                    incr = 2
                elif mode == "z":  # closing "z" or "Z"
                    incr = 1
                    svg_coords = np.append(svg_coords, d_line_origin, axis=0)
                else:
                    incr = 1
                    
                ii += incr
                continue
            
            else: 
                xy = chunk.split(',')
                xy = [float(s) for s in xy]
                
                # Deal with horizontal and vertical lines. 
                # They have only one number, so we have to fill the other with 0.0. 
                if mode == "h" and len(xy)==1:
                    if absolute: 
                        xy = [xy[0],current_point[1]]
                    else:
                        xy = [xy[0],0.0]
                elif mode == "v" and len(xy)==1: 
                    if absolute: 
                        xy = [current_point[0],xy[0]]
                    else:
                        xy = [0.0,xy[0]]
                
                # Now either initiate or append the new coordinate. 
                if first_absolute:
                    # Retract, then move, then set down at new point
                    if len(svg_coords) > 0:
                        svg_coords = np.append(svg_coords, [[current_point[0],current_point[1],0.05]],axis=0)
                    svg_coords = np.append(svg_coords, [[xy[0],xy[1],0.05]],axis=0)                
                    svg_coords = np.append(svg_coords, [[xy[0],xy[1],0.]],axis=0)
                    # set first_absolute to zero (not anymore). 
                    first_absolute = 0
                    d_line_origin = np.array([[xy[0],xy[1],0.0]])
                else:
                    if absolute:
                        svg_coords = np.append(svg_coords, [[xy[0],xy[1],0.]],axis=0)
                    else:
                        svg_coords = np.append(svg_coords, [current_point + np.array([xy[0],xy[1],0.])],axis=0)
                
                current_point = svg_coords[-1]
                
                if ii < len(chunks)-1:
                    if chunks[ii+1].isalpha():
                        incr = 1
                
                ii += incr

    svg_coords = np.append(svg_coords, [current_point + np.array([0.,0.,0.05])], axis=0)
                
    return svg_coords

def scale_coords_to_arena(coords, dx=dx , dy=dy, x_min=x_min, z_min=z_min):
    """
    Scales the coordinates from SVG to a specific size of workspace/arena 
    (rectangle, dimensions dx and dy)            
    """
    xsvg = coords[:,0]
    ysvg = coords[:,1]  # Note that Y is Down in SVG!
    zdesigned = coords[:,2]
    
    xmin = np.min(xsvg)
    xmax = np.max(xsvg)
    xrng = xmax - xmin
    xctr = xmin + xrng/2
    ymin = np.min(ysvg)
    ymax = np.max(ysvg)
    yrng = ymax - ymin
    yctr = ymin + yrng/2
 
    # Shift and scale. Also Rearrange Coordinates
    k = min([dx/xrng, dy/yrng])
    xscaled = -1*(ysvg-yctr)*k   # Note that Y is Down in SVG!
    yscaled = -1*(xsvg-xctr)*k
    xscaled = xscaled - np.min(xscaled) + x_min
    zscaled = zdesigned + z_min
    scaled_coords = np.vstack((xscaled,yscaled,zscaled)).transpose()
    return scaled_coords
    
    


if __name__=='__main__':

    svg_file = 'svg/swirl.svg'

    svg_coords = parse_svg_for_paths(svg_file)
    scaled_coords = scale_coords_to_arena(svg_coords)


    