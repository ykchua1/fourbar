#animates using function methods

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ani = 0
linkage = 0

def main():
    global linkage
        
    x, y = linkage.get_coord_array(linkage)
    
    x1, x2, x3, x4 = x[:,0], x[:,1], x[:,2], x[:,3]
    y1, y2, y3, y4 = y[:,0], y[:,1], y[:,2], y[:,3]
    
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    
    for i in tuple(x1)+tuple(x2)+tuple(x3)+ tuple(x4):
        if i < xmin:
            xmin = i
        if i > xmax:
            xmax = i
    for i in tuple(y1)+tuple(y2)+tuple(y3)+ tuple(y4):
        if i < ymin:
            ymin = i
        if i > ymax:
            ymax = i
    
    mar = .2*xmax #plot margin
            
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, 
                         xlim=(xmin-mar, xmax+mar), ylim=(ymin-mar, ymax+mar))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)
    
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        thisx = [x1[i], x2[i], x3[i], x4[i], x1[i]]
        thisy = [y1[i], y2[i], y3[i], y4[i], y1[i]]
    
        line.set_data(thisx, thisy)
        return line,
    
    global ani
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x1)),
                                  interval=20, blit=True, init_func=init)
    
    plt.show()
    
def anim_linkage(linkage_obj):
    global linkage
    linkage = linkage_obj
    
    main()
    
if __name__ == '__main__':
    main()