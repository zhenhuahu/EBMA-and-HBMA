import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# change yuv to rgb
def yuv2bgr(yuv_in):
	yuv_in[:, :, 0] = yuv_in[:, :, 0] - 16   # offset Y by 16
	yuv_in[:, :, 1:] = yuv_in[:, :, 1:] - 128  # offset UV by 128

	# YUV conversion matrix
	M = np.array([[1.164, 2.017, 0],
					[1.164, -0.392, -0.813],
					[1.164, 0,  1.596]])

	# to produce BGR output
	bgr_out = yuv_in.dot(M.T)

	return bgr_out


# YUV is the yuv data of the frame
# frameName is the name of the frame
def displayFrame(bgr_in, frameName, save_dir = ''):
		bgr_in = bgr_in.clip(0,255).astype(np.uint8)

		if len(save_dir) > 0:
			cv2.imwrite(save_dir, bgr_in)
			
		# display image with openCV
		cv2.imshow(frameName, bgr_in)

		key = cv2.waitKey(5000) # by default, wait for 5 seconds
		# quit display if space key is pressed
		if key == ord(' '):
			cv2.destroyAllWindows()

		cv2.destroyAllWindows()

		return 0

		#cv2.waitKey(0) # wait until any key is pressed
	
# draw motion field
def draw_motion_field(u, v, width, height, save_dir = ''):
	u,v = u/5, v/5
	s = u.shape
	print(s)
	x,y = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
	plt.figure(1)
	q = plt.quiver(x, y, u, v, scale=1, scale_units='xy', pivot = 'tail', angles = 'xy')
	if len(save_dir) == 0:
		plt.show()
	else:
		plt.savefig(save_dir)
		plt.close()


# calcuate the psnr
def psnr(img1, img2):
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	else:
		PIXEL_MAX = 255.0
		return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

