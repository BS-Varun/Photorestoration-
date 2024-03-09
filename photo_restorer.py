from dotenv import load_dotenv
load_dotenv()
import replicate
import numpy as np
from PIL import Image
import imageio
# Define the mask_var variable or import it from another module
# For example, assuming it's an array of zeros and ones:
mask_var = np.array([[0, 1], [1, 0]])
# from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from gfpgan import GFPGANer
# colorizer = get_image_colorizer(artistic=True)
def inpaint_and_restore_with_mask(img_path, mask_path):
    # Load the image and mask
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Perform inpainting (replace this with your inpainting logic)
    inpainted_region = inpaint(img, mask)

    # Perform restoration using GFPGAN (replace this with your restoration logic)
    restored_img = restore(inpainted_region)

    # Pass the restored image through deoldify
    deoldified_img = deoldify_image(restored_img)

    # Save the inpainted, restored, and deoldified images
    inpainted_path = 'static/images/inpainted_' + secure_filename(os.path.basename(img_path))
    restored_path = 'static/images/restored_' + secure_filename(os.path.basename(img_path))
    deoldified_path = 'static/images/deoldified_' + secure_filename(os.path.basename(img_path))
    
    cv2.imwrite(inpainted_path, inpainted_region)
    cv2.imwrite(restored_path, restored_img)
    cv2.imwrite(deoldified_path, deoldified_img)

    return inpainted_path, restored_path, deoldified_path

# Define the inpaint and restore functions
def inpaint(img, mask):
    inpainted_region = img.copy()
    inpainted_region[mask != 0] = [255, 255, 255]  # Set masked region to white
    return inpainted_region

def restore(inpainted_region):
    # Replace this with your restoration logic
    gfpgan_model_path = 'path/to/your/GFPGAN/model.pth'  # Replace with your actual GFPGAN model path
    gfpgan = GFPGANer(gfpgan_model_path)
    restored_img = gfpgan.enhance(inpainted_region)
    return restored_img

# Define the inpainted_region function
def get_inpainted_region(mask_var):
    inpainted_region = mask_var  # Replace this with your logic
    return inpainted_region

# Rest of the code...

# Assuming these are defined elsewhere in your code
net_input_saved = ...
noise = ...

def closure():
    global i, inpainted

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    if inpainted:
        out = inpainting_scale * out + (1 - inpainting_scale) * inpainted_region

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')

    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        if img_np.shape[0] == 3:
            plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        else:
            plot_image_grid([np.clip(out_np[0], 0, 1)], factor=figsize, nrow=1)

    inpainted = out * (1 - mask_var) + inpainted_region

    i += 1
    return total_loss

# ... (remaining code)
def is_black_and_white(image_path):
    image = imageio.imread(image_path)

    # Check if the image has a single channel (grayscale)
    is_bw = len(image.shape) == 2

    return is_bw


# Define the deoldify_image function
# def deoldify_image(image):
#     deoldified_img = colorize_image(image)
#     return deoldified_img

# Define the predict_img function
def predict_img(filename):
    output = replicate.run(
        "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
        input={"img": open(filename, "rb")}
    )
    print(output)
    return output

def deoldify(filename1):
   print(filename1)
   output1 = replicate.run(
  "arielreplicate/deoldify_image:0da600fab0c45a66211339f1c16b71345d22f26ef5fea3dca1bb90bb5711e950",
  input={
    "model_name": "Artistic",
    "input_image": filename1,
    "render_factor": 40
    }
   )
   print(output1)
   return output1

# def deoldify1(filename1):
#    output2 = replicate.run(
#        "arielreplicate/deoldify_image:0da600fab0c45a66211339f1c16b71345d22f26ef5fea3dca1bb90bb5711e950",
#        input={"input_image": open(filename1, "rb")}
#    )
#    print(output2)
#    return output2


# gfpgan_output = predict_img(filename)  # Replace with the actual image path
