# 2 layers of hbma

import numpy as np
import cv2

from getYuvFrame import getYUVFrame
from display import displayFrame, yuv2bgr
from display import draw_motion_field, psnr 
import time

# exhaustive block matching algorithm
class hbma():
	# N: block size; R: search range
	def __init__(self, video, N, R):
		self.video = video
		self.N = N   # block size
		self.R = R  # search range

	def match(self):
		# video frame width and height
		start_time = time.time()
		width = 352
		height = 288
		N = self.N

		search_params = '_'+str(N)+'_'+str(self.R)+'.jpg'

		frames = getYUVFrame(self.video, width, height)
		anchor = yuv2bgr(frames.getFrame(6))  # anchor frame is frame 6 
		#print(anchor.shape)
		#print(anchor)
		target = yuv2bgr(frames.getFrame(22))  # 22 target frame is frame 22

		#displayFrame(anchor, 'anchor')
		#displayFrame(target, 'target')

		# downsample the anchor image by 2
		anchor_small = cv2.resize(anchor, (0,0), fx = 0.5, fy = 0.5)
		#print(anchor_small.shape)

		# downsample the targte image by 2
		target_small = cv2.resize(target, (0,0), fx = 0.5, fy = 0.5)

		width_small = anchor_small.shape[1]
		#print(width_small)
		height_small = anchor_small.shape[0]

		# use ebma to find the predicted small block
		predict_small = np.zeros(anchor_small.shape)

		R_small = int(np.floor(self.R/(2)))
		d_small = np.maximum(N, R_small)

		# padding 0's for further processing
		anchor_small_2 = np.pad(anchor_small, ((d_small,d_small),(d_small,d_small),(0,0)), 'constant', constant_values = 0) # pad the anchor frame with 0 
		target_small_2 = np.pad(target_small, ((d_small,d_small),(d_small,d_small),(0,0)), 'constant', constant_values = 0) # target

		# average all channels into 1
		f1_small = anchor_small_2.mean(2) 
		f2_small = target_small_2.mean(2)
		#print('f2_small.shape is {}'.format(f2_small.shape))

		numWidthBlks_small = int(np.ceil(width_small / N))
		numHeightBlks_small = int(np.ceil(height_small / N))

		# store MV image
		mvx_small = np.ones([numHeightBlks_small, numWidthBlks_small])
		mvy_small = np.ones([numHeightBlks_small, numWidthBlks_small])

		# use ebma to find matches
		for ii in range(d_small, d_small-1 + height_small, N):
			for jj in range(d_small, d_small-1 + width_small, N): # every block in the anchor frame
				MAD_min = 256*N*N

				for kk in range(-R_small, R_small+1):
					for ll in range(-R_small, R_small+1): # every search candidate
						#print(ii+kk+N)
						MAD = np.sum(np.absolute(f1_small[ii: ii+N, jj:jj+N] - f2_small[ii+kk: ii+kk+N, jj+ll:jj+ll+N]))

						if MAD < MAD_min:
							MAD_min = MAD
							# memorize the 
							dy = kk
							dx = ll

				# put the best matching block in the predicted image
				predict_small[ii-d_small: ii-d_small+N, jj-d_small: jj-d_small+N, :] = target_small_2[ii+dy: ii+dy+N, jj+dx: jj+dx+N, :]
				
				# record the estimated MV in a matrix
				iblk = int(np.floor((ii-d_small-1)/N)+1)
				#print(iblk)
				jblk = int(np.floor((jj-d_small-1)/N+1))

				mvx_small[iblk, jblk] = dx
				mvy_small[iblk, jblk] = dy

		# original layer
		# padding 0's for further processing
		anchor_2 = np.pad(anchor, ((d_small,d_small),(d_small,d_small),(0,0)), 'constant', constant_values = 0) # pad the anchor frame with 0 
		target_2 = np.pad(target, ((d_small,d_small),(d_small,d_small),(0,0)), 'constant', constant_values = 0) # target

		# average all channels into 1 
		f1 = anchor_2.mean(2)
		f2 = target_2.mean(2)

		numWidthBlks = int(np.ceil(width / N))
		numHeightBlks = int(np.ceil(height / N))

		# store MV image
		# enlarge the small MV's by 2*2
		mvx_small_repl = np.ones([numHeightBlks, numWidthBlks])
		mvy_small_repl = np.ones([numHeightBlks, numWidthBlks])

		for i in range(numHeightBlks_small):
			for j in range(numWidthBlks_small):
				mvx_small_repl[i*2: i*2+2, j*2: j*2+2] = mvx_small[i,j]
				mvy_small_repl[i*2: i*2+2, j*2: j*2+2] = mvy_small[i,j]

		# store MVs of original image
		mvx = np.ones([numHeightBlks, numWidthBlks])
		mvy = np.ones([numHeightBlks, numWidthBlks])

		predict = np.zeros(target.shape)

		for ii in range(d_small, d_small-1 + height, N):
			for jj in range(d_small, d_small-1 + width, N): # every block in the anchor frame
				MAD_min = 256*N*N
				j_idx = int(mvx_small_repl[int(np.floor((ii-d_small)/N)),int(np.floor((jj-d_small)/N))] *2)
				i_idx = int(mvy_small_repl[int(np.floor((ii-d_small)/N)),int(np.floor((jj-d_small)/N))]*2)

				for kk in range(-R_small, R_small+1):
					for ll in range(-R_small, R_small+1): # every search candidate
						MAD = np.sum(np.absolute(f1[ii: ii+N, jj:jj+N] - f2[ii+i_idx+kk: ii+i_idx+kk+N, jj+j_idx+ll:jj+j_idx+ll+N]))

						if MAD < MAD_min:
							MAD_min = MAD
							# memorize the 
							dy = kk
							dx = ll

							#print('{}:{} -> {}'.format(dy,dx, MAD_min))
				# put the best matching block in the predicted image
				predict[ii-d_small: ii-d_small+N, jj-d_small: jj-d_small+N, :] = target_2[ii+dy: ii+dy+N, jj+dx: jj+dx+N, :]
				
				# record the estimated MV in a matrix
				iblk = int(np.floor((ii-d_small-1)/N)+1)
				#print(iblk)
				jblk = int(np.floor((jj-d_small-1)/N+1))

				mvx[iblk, jblk] = dx
				mvy[iblk, jblk] = dy


		# error image between target and predicted

		process_time = time.time() - start_time
		print('processing time is {}'.format(process_time))

		error_img = target - predict
		error_img = error_img.clip(min = 0)

		print('psnr is {}'.format(psnr(target, predict)))

		displayFrame(predict, 'predict', 'hbma_predict' + search_params)
		displayFrame(error_img, 'error', 'hbma_error'+ search_params)

		#print(mvx)
		# draw the motion field
		draw_motion_field(mvx, mvy, width, height, 'hbma_mv'+search_params)

		return mvx, mvy










