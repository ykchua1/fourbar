import objects.fourbar as fourbar
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt

linkage = fourbar.FourBar(1,2,3,3.5,0)
deg = np.array([])
mech_adv = np.array([])
R = np.array([[cos(np.pi/2), -sin(np.pi/2)],
              [sin(np.pi/2), cos(np.pi/2)]])
for i in range(120):
    loc_in = linkage.get_coords2(linkage)[1]
    loc_out = linkage.get_coords2(linkage)[2]
    crank_vec = np.array(linkage.get_coords2(linkage)[1]) - \
                np.array(linkage.get_coords2(linkage)[0])
    f_in = np.matmul(R, crank_vec) / np.linalg.norm(crank_vec)
    x = linkage.get_inst_moment2(linkage, f_in, loc_in)
    y = linkage.get_inst_moment3(linkage, (0, 1), loc_out)
    
    deg = np.append(deg, linkage.crank_angle)
    mech_adv = np.append(mech_adv, x/y)
    
    linkage.turn_crank2(linkage, 3)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlabel='crank angle (deg)', 
       ylabel='mechanical advantage (F_out/F_in)',
       title='mechanical advantage vs crank angle')
ax.set_yticks([n for n in range(-30, 30, 5)])
ax.minorticks_on()
ax.grid(which='both')
ax.set_ylim(-20, 20)
ax.plot(deg, mech_adv, color='red')
plt.show()