import matplotlib.pyplot as plt
import matplotlib.animation as animation

ani = 0
linkage = 0

def main():
    global linkage
        
    gen = linkage.get_coord_gen(linkage)
    
    lengths = [linkage.crank, linkage.coupler, 
               linkage.follower, linkage.ground]
    lengths.sort(reverse=True)
    ori = linkage.origin
    lim = lengths[0] + lengths[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, 
                         xlim=(ori[0]-lim, ori[0]+lim), 
                         ylim=(ori[1]-lim, ori[1]+lim))
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)
    
    def animate(i):
        x, y = i
        thisx = [x[0,0], x[0,1], x[0,2], x[0,3], x[0,0]]
        thisy = [y[0,0], y[0,1], y[0,2], y[0,3], y[0,0]]

        line.set_data(thisx, thisy)
        return line,

    interval = 5000/360
    
    global ani
    ani = animation.FuncAnimation(fig, animate, gen, save_count=360,
                                  interval=interval, blit=True)
    
    plt.show()
    
def anim_linkage(linkage_obj):
    global linkage
    linkage = linkage_obj
    
    main()
    
if __name__ == '__main__':
    main()