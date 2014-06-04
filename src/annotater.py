import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import Image, ImageDraw
from pylab import subplot, show
import numpy as np

def toggle_selector(event):
	# we need this, but I do not know why
	pass



class Annotater(object):

	def __init__(self):
		self.rectangles = []

	def onselect(self, eclick, erelease):
		"""eclick and erelease are matplotlib events at press and release"""

		#print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
		#print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
		#print ' used button   : ', eclick.button

		if eclick.button == 3:
			if len(self.rectangles) > 0:
				self.rectangles = self.rectangles[:-1]

		#self.ax.set_ylim(erelease.ydata,eclick.ydata)
		#self.ax.set_xlim(eclick.xdata,erelease.xdata)
		#fig.canvas.draw()

		draw = ImageDraw.Draw(self.image_plot)
		draw.line((0, 0, 100, 100), fill=128)
		self.image.show()

	def test(self, folder='../data/test/'):
		for i in range(1, 4):

			# reset rectangles
			self.rectangles = []

			f = '%s500_test%d.jpg' % (folder, i)
			print f

			#if f[:4] != '500_' or f[-4:] != '.jpg':
			#	continue

			# open image and convert to np array
			self.image = Image.open(f)
			image_array = np.asarray(self.image)

			# create figure handlers
			fig = plt.figure()
			self.ax = fig.add_subplot(111)
			self.image_plot = plt.imshow(image_array)


			toggle_selector.RS = widgets.RectangleSelector(self.ax, self.onselect,
				drawtype='box',
				rectprops = dict(facecolor='red', edgecolor = 'black',
				alpha=0.5, fill=True))
			#connect('key_press_event', toggle_selector)
			show()

if __name__ == '__main__':
	annotater = Annotater()
	annotater.test()
