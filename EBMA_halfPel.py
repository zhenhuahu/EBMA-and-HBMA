import numpy as np
import cv2

from getYuvFrame import getYUVFrame
from display import displayFrame, yuv2bgr
from display import draw_motion_field, psnr 
import time


# exhaustive block matching algorithm
class ebma_halfPel():
	# N: block size; R: search range
	def __init__(self, video, N, R):
		self.video = video
		self.N = N * 2   # block size
		self.R = R * 2  # search range

	def match(self):
		# video frame width and height
		start_time = time.time()
		width = 352
		height = 288
		N = self.N

		search_params = '_'+str(int(N/2))+'_'+str(int(self.R/2))+'.jpg'

		frames = getYUVFrame(self.video, width, height)
		anchor_ori = yuv2bgr(frames.getFrame(6))  # anchor frame is frame 6 
		# upsample image using bilinear interpolation
		anchor = cv2.resize(anchor_ori, (width*2, height*2))

		#print(anchor.shape)
		#print(anchor)
		target_ori = yuv2bgr(frames.getFrame(22))  # 22 target frame is frame 22
		target = cv2.resize(target_ori, (width*2, height*2))

		#displayFrame(anchor_ori, 'anchor')
		#displayFrame(target_ori, 'target')

		predict = np.zeros(anchor.shape)
		#displayFrame(predict, 'predict1')

		d = np.maximum(self.N, self.R)

		# padding 0's for further processing
		anchor_2 = np.pad(anchor, ((d,d),(d,d),(0,0)), 'constant', constant_values = 0) # pad the anchor frame with 0 
		target_2 = np.pad(target, ((d,d),(d,d),(0,0)), 'constant', constant_values = 0) # target

		# average all channels into 1 
		f1 = anchor_2.mean(2)
		f2 = target_2.mean(2)

		width = width*2
		height = height*2

		numWidthBlks = int(np.ceil(width / N))
		numHeightBlks = int(np.ceil(height / N))
		#print(numWidthBlks, numHeightBlks)

		# store MV image
		mvx = np.ones([numHeightBlks, numWidthBlks])
		mvy = np.ones([numHeightBlks, numWidthBlks])

		for ii in range(d, d-1 + height, N):
			for jj in range(d, d-1 + width, N): # every block in the anchor frame
				MAD_min = 256*N*N

				for kk in range(-self.R, self.R+1):
					for ll in range(-self.R, self.R+1): # every search candidate
						MAD = np.sum(np.absolute(f1[ii: ii+N, jj:jj+N] - f2[ii+kk: ii+kk+N, jj+ll:jj+ll+N]))

						if MAD < MAD_min:
							MAD_min = MAD
							# memorize the 
							dy = kk
							dx = ll

							#print('{}:{} -> {}'.format(dy,dx, MAD_min))

				# put the best matching block in the predicted image
				predict[ii-d: ii-d+N, jj-d: jj-d+N, :] = target_2[ii+dy: ii+dy+N, jj+dx: jj+dx+N, :]
				
				# record the estimated MV in a matrix
				iblk = int(np.floor((ii-d-1)/N)+1)
				#print(iblk)
				jblk = int(np.floor((jj-d-1)/N+1))

				mvx[iblk, jblk] = dx
				mvy[iblk, jblk] = dy

		#print(predict.shape)

		predict = cv2.resize(predict, (0,0), fx = 0.5, fy = 0.5)

		process_time = time.time() - start_time

		print('processing time is {}'.format(process_time))

		displayFrame(predict, 'predict', 'ebma_halfPel_predict' + search_params)

		# error image between target and predicted
		error_img = target_ori - predict
		error_img = error_img.clip(min = 0)

		displayFrame(error_img, 'error', 'ebma_halfPel_error' + search_params)

		print('psnr is {}'.format(psnr(target_ori, predict)))

		#print(mvx)
		# draw the motion field
		draw_motion_field(mvx, mvy, width, height, 'ebma_halfPel_mv' + search_params)

		return mvx, mvy










