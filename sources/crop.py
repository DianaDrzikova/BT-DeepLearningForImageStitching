import numpy as np

# Diana Maxima Drzikova

def crop(image, x, y, angle):

    found = False
    left, right, top, bottom = 0,0,0,0

    # rows
    for y1 in range(y): 
        # columns
        for column in range(x): 
            # first found non-null element
            if image[y1][column]:
                if angle < 0: # rotation right indicates search for left edge
                    left = column
                else: # rotation left indicates search for right edge
                    right = column   
                found = True
                break
        if found:
            break
    
    found = False   

    # columns 
    for x1 in range(x,0,-1):
        # rows
        for row in range(y):
            if image[row][x1]:
                if angle < 0: # rotation right indicates search for top edge
                    top = row
                else: # rotation left indicates search for bottom edge
                    bottom = row
                found = True
                break
        if found:
            break

    # based on triangle and rectangle axioms
    if angle < 0:
        right = x-left
        bottom = y-top
    else:
        left = x-right
        top = y-bottom

    #print(left, right, top, bottom)
    return left, top, right, bottom

def anglecut(angle, x, y, xp, yp):

    max_angle = 90
 
    ratio = np.abs(angle/max_angle)

    top = int(ratio*y)
    bottom = y - int(ratio*y)
    left = top*2
    right = x - left

    return left, top, right, bottom