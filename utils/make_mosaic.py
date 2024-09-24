import numpy as np
import math

def find_rectangle(n, max_ratio=2):
    sides = []
    square = int(math.sqrt(n))
    for w in range(square, max_ratio * square):
        h = n / w
        used = w * h
        leftover = n - used
        sides.append((leftover, (w, h)))
    return sorted(sides)[0][1]

def make_mosaic(images, n=None, nx=None, ny=None, w=None, h=None):
    print("Let's make a MOSAIC!")
    if n is None and nx is None and ny is None:
        nx, ny = find_rectangle(len(images))
    else:
        nx = n if nx is None else nx
        ny = n if ny is None else ny
    print(nx, ny)
    
    images = np.array(images)
    if images.ndim == 2:
        side = int(np.sqrt(len(images[0])))
        h = side if h is None else h
        w = side if w is None else w
        images = images.reshape(-1, h, w)
    else:
        h = images.shape[1]
        w = images.shape[2]
    
    print(f"h, w is {h}, {w}")
    print(f"images has len {len(images)}")
    
    image_gen = iter(images)
    mosaic = np.empty((h * ny, w * nx))
    
    for i in range(ny):
        ia = i * h
        ib = (i + 1) * h
        for j in range(nx):
            ja = j * w
            jb = (j + 1) * w
            print(f"ia, ib {ia}, {ib}  ja, jb {ja}, {jb}")
            try:
                mosaic[ia:ib, ja:jb] = next(image_gen)
            except StopIteration:
                print("No more images available to complete the mosaic.")
                break
            except Exception as e:
                print(f"Unexpected exception: {e}")
    
    return mosaic
