#draws linkage

import matplotlib.pyplot as plt

linkage = 0

def main():
    global linkage
    
    x = [linkage.get_coords2(linkage)[0][0],
         linkage.get_coords2(linkage)[1][0],
         linkage.get_coords2(linkage)[2][0],
         linkage.get_coords2(linkage)[3][0],
         linkage.get_coords2(linkage)[0][0]]
    y = [linkage.get_coords2(linkage)[0][1],
         linkage.get_coords2(linkage)[1][1],
         linkage.get_coords2(linkage)[2][1],
         linkage.get_coords2(linkage)[3][1],
         linkage.get_coords2(linkage)[0][1]]
    
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    
    for i in x:
        if i < xmin:
            xmin = i
        if i > xmax:
            xmax = i
    for i in y:
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
    
    ax.plot(x, y, 'o-', lw=2)

    plt.show()
    
def plt_linkage(linkage_obj):
    global linkage
    linkage = linkage_obj
    
    main()
    
if __name__ == '__main__':
    main()