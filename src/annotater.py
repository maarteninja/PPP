import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import Image
from pylab import subplot, show
import numpy as np

def toggle_selector(event):
    # we need this, but I do not know why
    pass



class Annotater(object):

    def onselect(self, eclick, erelease):
        """eclick and erelease are matplotlib events at press and release"""

        print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
        print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
        print ' used button   : ', eclick.button

        if eclick.button == 3:
            print 'shall remove?'

    def test(self, folder='../data/test/'):
        for i in range(1, 4):
            f = '%s500_test%d.jpg' % (folder, i)
            print f

            #if f[:4] != '500_' or f[-4:] != '.jpg':
            #    continue

            # open image and convert to np array
            image = Image.open(f)
            image_array = np.asarray(image)

            # create figure handlers
            fig = plt.figure()
            ax = fig.add_subplot(111)

            image_plot = plt.imshow(image_array)

            #widgets.RectangleSelector(ax, onselect, drawtype='box')

            toggle_selector.RS = widgets.RectangleSelector(ax, self.onselect, drawtype='box')
            #connect('key_press_event', toggle_selector)
            show()

if __name__ == '__main__':
    annotater = Annotater()
    annotater.test()
