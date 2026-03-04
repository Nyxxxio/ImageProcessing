import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread(r"C:\Users\rithw\uniiiiiiiiiii\Image Proc\HW2\grayish.jpg")

b,g,r=cv2.split(img)

gamma=0.6
b_gamma=255*((b/255)**gamma)
g_gamma=255*((g/255)**gamma)
r_gamma=255*((r/255)**gamma)

gamma_img=cv2.merge((b_gamma.astype(np.uint8),g_gamma.astype(np.uint8),r_gamma.astype(np.uint8)))

b_minmax=255*(b-np.min(b))/(np.max(b)-np.min(b))
g_minmax=255*(g-np.min(g))/(np.max(g)-np.min(g))
r_minmax=255*(r-np.min(r))/(np.max(r)-np.min(r))

minmax_img=cv2.merge((b_minmax.astype(np.uint8),g_minmax.astype(np.uint8),r_minmax.astype(np.uint8)))

plt.figure(figsize=(10,4))
plt.subplot(131);plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(132);plt.imshow(cv2.cvtColor(gamma_img,cv2.COLOR_BGR2RGB))
plt.subplot(133);plt.imshow(cv2.cvtColor(minmax_img,cv2.COLOR_BGR2RGB))
plt.show()

for i,c in enumerate(['r','g','b']):
    plt.plot(cv2.calcHist([img],[i],None,[256],[0,256]),c)
plt.show()

for i,c in enumerate(['r','g','b']):
    plt.plot(cv2.calcHist([gamma_img],[i],None,[256],[0,256]),c)
plt.show()

for i,c in enumerate(['r','g','b']):
    plt.plot(cv2.calcHist([minmax_img],[i],None,[256],[0,256]),c)
plt.show()
