from io import BytesIO
import numpy as np
from PIL import Image
import IPython.display
import imageio
import shutil

def show_array(a, fmt='png', filename=None):
    a = np.uint8(np.clip(a, 0, 255))
    print(f"a has type {type(a)}")
    
    image_data = BytesIO()
    print(f"write image 'output.png'")
    imageio.imwrite("output.png", a, format="PNG")
    
    image = Image.fromarray(a)
    image.save(image_data, fmt)
    
    if filename is None:
        IPython.display.display(IPython.display.Image(data=image_data.getvalue()))
    else:
        with open(filename, 'wb') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)
