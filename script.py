#script to animate linkage and saves as mp4 file

import objects.fourbar
from methods import plot, animate, animate2

linkage = objects.fourbar.FourBar(1,2,3,3.5)
plot.plt_linkage(linkage)
#animate2.anim_linkage(linkage)
animate.anim_linkage(linkage)

Writer = animate.animation.writers['ffmpeg']
writer = Writer(fps=60, bitrate=500)
animate.ani.save('movie2.mp4', writer=writer)