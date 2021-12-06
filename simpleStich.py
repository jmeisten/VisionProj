from PIL import Image
import os

# Open images and store them in a list
images = [Image.open('/home/pi/testpics/'+x) for x in os.listdir('/home/pi/testpics/')]

# might want to chop the images using:
# Image.crop(left, top, right, bottom)

# or pipe images through a ML so it chops it for us 
# object detection????

total_width = 0
max_height = 0
print(images)
# find the width and height of the final image
for img in images:
    total_width += img.size[0]
    max_height = max(max_height, img.size[1])

# create a new image with the appropriate height and width
new_img = Image.new('RGB', (total_width, max_height))
# Write the contents of the new image
current_width = 0
for img in images:
  new_img.paste(img, (current_width,0))
  current_width += img.size[0]
# Save the image
new_img.save('NewImage.jpg')