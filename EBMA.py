import numpy as np

from getYuvFrame import getYUVFrame
from display import displayFrame
from display import draw_motion_field, psnr 
from display import yuv2bgr
import time


# exhaustive block matching algorithm
class ebma():
	# N: block size; R: search range
	def __init__(self, video, N, R):
		self.video = video
		self.N = N   # block size
		self.R = R  # search range

	def match(self):
		start_time = time.time()
		# video frame width and height
		width = 352
		height = 288
		N = self.N
		search_params = '_'+str(N)+'_'+str(self.R)+'.jpg'

		frames = getYUVFrame(self.video, width, height)
		anchor = yuv2bgr(frames.getFrame(6))  # anchor frame is frame 6 
		#print(anchor.shape)
		#print(anchor)
		target = yuv2bgr(frames.getFrame(22))  # 22 target frame is frame 22

		#displayFrame(anchor, 'anchor', 'anchor.jpg')
		#displayFrame(anchor2, 'anchor2')
		#displayFrame(target, 'target', 'target.jpg')

		predict = np.zeros(anchor.shape)
		#displayFrame(predict, 'predict1')

		d = np.maximum(self.N, self.R)

		# padding 0's for further processing
		anchor_2 = np.pad(anchor, ((d,d),(d,d),(0,0)), 'constant', constant_values = 0) # pad the anchor frame with 0 
		target_2 = np.pad(target, ((d,d),(d,d),(0,0)), 'constant', constant_values = 0) # target

		f1 = anchor_2.mean(2)
		f2 = target_2.mean(2)

		#print('f1 shape is {}'.format(f1.shape))

		#print('anchor_2 shape is {}'.format(anchor_2.shape))
		numWidthBlks = int(np.ceil(width / N))
		numHeightBlks = int(np.ceil(height / N))
		#print(numWidthBlks, numHeightBlks)

		# store MV image
		mvx = np.zeros([numHeightBlks, numWidthBlks])
		mvy = np.zeros([numHeightBlks, numWidthBlks])

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

				# put the best matching block in the predicted image
				predict[ii-d: ii-d+N, jj-d: jj-d+N, :] = target_2[ii+dy: ii+dy+N, jj+dx: jj+dx+N, :]
				
				# record the estimated MV in a matrix
				iblk = int(np.floor((ii-d-1)/N)+1)
				#print(iblk)
				jblk = int(np.floor((jj-d-1)/N+1))

				mvx[iblk, jblk] = dx
				mvy[iblk, jblk] = dy

		#print('predict shape is {}'.format(predict.shape))
		process_time = time.time() - start_time
		print('processing time is {}'.format(process_time))
		displayFrame(predict, 'predict', 'ebma_predict'+search_params)

		# error image between target and predicted
		error_img = target - predict
		error_img = error_img.clip(min = 0)

		print('psnr is {}'.format(psnr(target, predict)))
		
		displayFrame(error_img, 'error', 'ebma_error'+search_params)


		#print(mvx)
		# draw the motion field
		draw_motion_field(mvx, mvy, width, height, 'ebma_mv'+search_params)

		return mvx, mvy










