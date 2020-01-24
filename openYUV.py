import numpy as np
import cv2


width = 352
height = 288

stream = open('akiyo_352x288_30.yuv','rb')

# seek to the fourth frame in the file
stream.seek(4*width*height*1.5)

# calculate the actual image size in the stream
fwidth = (width + 31) // 32 *32
fheight = (height + 15) // 16 * 16

# load the Y (luminance) data from the stream
Y = np.fromfile(stream, dtype = np.uint8, count = fwidth*fheight).reshape(fheight, fwidth)

# load the UV (chrominance) data from the stream, and double its size
U = np.fromfile(stream, dtype = np.uint8, count = (fwidth//2)*(fheight//2)).\
	reshape((fheight//2, fwidth//2)).repeat(2, axis=0).repeat(2, axis=1)

U = np.fromfile(stream, dtype = np.uint8, count = (fwidth//2)*(fheight//2)).\
	reshape((fheight//2, fwidth//2)).repeat(2, axis=0).repeat(2, axis=1)

# stack the YUV channels together, crop the actual resolution, convert to
# floating point for later calculations, and apply the standard biases
YUV = np.dstack((Y,U,V))[:height, :width, :].astype(np.float)
YUV[:, :, 0] = YUV[:, :, 0] - 16   # offset Y by 16
YUV[:, :, 1] = YUV[:, :, 1] - 128  # offset UV by 128

# YUV conversion matrix
M = np.array([[1.164, 2.017, 0],
				[1.164, -0.392, -0.813],
				[1.164, 0,  1.596]])

# to produce BGR output
BGR = YUV.dot(M.T).clip(0,255).astype(np.uint8)

# display image with openCV
cv2.imshow('image', BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

