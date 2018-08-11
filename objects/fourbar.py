class FourBar():

    import numpy as np

    def __init__(self, crank, coupler, follower, ground,
                 crank_angle=90, grd_angle=0, typ='convex', origin=(0,0)):
        self.origin = origin
        self.crank = crank
        self.coupler = coupler
        self.follower = follower
        self.ground = ground
        self.crank_angle = crank_angle
        self.grd_angle = grd_angle
        self.typ = typ

    @property #1
    def crank(self):
        return self._crank
    @crank.setter
    def crank(self, value):
        self._crank = value
    @property #2
    def coupler(self):
        return self._coupler
    @coupler.setter
    def coupler(self, value):
        self._coupler = value
    @property #3
    def follower(self):
        return self._follower
    @follower.setter
    def follower(self, value):
        self._follower = value
    @property #4
    def ground(self):
        return self._ground
    @ground.setter
    def ground(self, value):
        self._ground = value
    @property #5
    def crank_angle(self):
        return self._crank_angle
    @crank_angle.setter
    def crank_angle(self, value):
        if value < 0 and value >= -360:
            value = 360 + value
        elif value > 360 and value <= 720:
            value -= 360
        elif value < -360 or value >720:
            raise ValueError
        self._crank_angle = value
    @property #6
    def grd_angle(self):
        return self._grd_angle
    @grd_angle.setter
    def grd_angle(self, value):
        if value < 0 and value >= -360:
            value = 360 + value
        elif value > 360 and value <= 720:
            value -= 360
        elif value < -360 or value >720:
            raise ValueError
        self._grd_angle = value
    @property #7
    def typ(self):
        return self._typ
    @typ.setter
    def typ(self, value):
        if not self.isBroken() and self.is_typ_valid(self, value):
            self._typ = value
        else:
            raise ValueError
    @property #8
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self, value):
        if len(value) == 2:
            self._origin = tuple(value)
        else:
            raise ValueError

    def get_state(self, state=''): #choose the state (crank? coupler?)
        ans = (self.crank, self.coupler, self.follower, self.ground,
               self.crank_angle, self.grd_angle, self.typ, self.origin)
        dic = {'crank':0, 'coupler': 1, 'follower': 2, 'ground': 3,
               'crank_angle': 4, 'grd_angle': 5, 'type': 6, 'origin': 7}
        if len(state)==0:
            return ans
        return ans[dic[state]]

    def set_state(self, crank='False', coupler='False', follower='False',
                  ground='False', c_angle='False', g_angle='False',
                  typ='False', origin='False'):

        typ = "'"+typ+"'" if typ!='False' else typ

        from copy import deepcopy
        copy = deepcopy(self)

        inp = (crank, coupler, follower, ground,
               c_angle, g_angle, typ, origin)
        out = ['crank', 'coupler', 'follower', 'ground',
               'crank_angle', 'grd_angle', 'typ', 'origin']

        for i in range(len(inp)):
            if inp[i] != 'False':
                exec('copy.'+out[i]+'={}'.format(inp[i]))

        if self.is_typ_valid(copy, copy.typ) and not copy.isBroken():
            for i in range(len(inp)):
                if inp[i] != 'False':
                    exec('self.'+out[i]+'={}'.format(inp[i]))
        else:
            raise ValueError

    def isBroken(self):
        alpha = abs(self.crank_angle-self.grd_angle)
        if self.coupler+self.follower<self.cos_law(self.crank, self.ground,
                                                   alpha):
            return True
        try:
            self.intersects(self.get_crank()[1], self.get_ground()[0],
                            self.coupler, self.follower)
        except:
            return True
        return False

    def intersects(self, A, B, r1, r2):
        x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
        d = self.np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        l = (r1**2 - r2**2 + d**2)/(2*d)
        h = self.np.sqrt(r1**2 - l**2)

        x_int1 = (l/d)*(x2 - x1) + (h/d)*(y2 - y1) + x1
        y_int1 = (l/d)*(y2 - y1) - (h/d)*(x2 - x1) + y1

        x_int2 = (l/d)*(x2 - x1) - (h/d)*(y2 - y1) + x1
        y_int2 = (l/d)*(y2 - y1) + (h/d)*(x2 - x1) + y1

        from math import isnan
        ans = ((x_int1, y_int1), (x_int2, y_int2))
        for i in ans:
            for j in i:
                if isnan(j):
                    raise ValueError
        return ans

    def get_crank(self):
        crank_angle_rad = self.crank_angle/180*self.np.pi
        x1, y1 = self.origin[0], self.origin[1]
        delta_x = self.crank*self.np.cos(crank_angle_rad)
        delta_y = self.crank*self.np.sin(crank_angle_rad)
        return ((x1,y1), (x1+delta_x, y1+delta_y))

    def get_ground(self):
        ground_angle_rad = self.grd_angle/180*self.np.pi
        x2, y2 = self.origin[0], self.origin[1]
        delta_x = self.ground*self.np.cos(ground_angle_rad)
        delta_y = self.ground*self.np.sin(ground_angle_rad)
        return ((x2+delta_x, y2+delta_y), (x2, y2))

    def get_coords(self, linkage, int_index):
        A = linkage.origin
        B = linkage.get_crank()[1]
        D = linkage.get_ground()[0]
        C = linkage.intersects(B, D, linkage.coupler,
                               linkage.follower)[int_index]
        return (A, B, C, D)

    def get_int_coord(self, linkage):
        lefts = self.left_turns(self.get_coords(linkage,0))
        if linkage.typ=='convex':
            if lefts==4 or lefts==0:
                return self.get_coords(linkage,0)[2]
            return self.get_coords(linkage,1)[2]
        elif linkage.typ=='concave':
            if lefts==3 or lefts==1:
                return self.get_coords(linkage,0)[2]
            return self.get_coords(linkage,1)[2]
        else:
            if lefts==2:
                return self.get_coords(linkage,0)[2]
            return self.get_coords(linkage,1)[2]

    def get_coords2(self, linkage):
        A = linkage.origin
        B = linkage.get_crank()[1]
        D = linkage.get_ground()[0]
        C = self.get_int_coord(linkage)
        return(A, B, C, D)

    #calculate turning angle
    def calc_angle(self, A, B, C):
        vec1 = self.np.array(B) - self.np.array(A)
        vec1_angle = self.np.angle(vec1[0] + vec1[1]*1j, deg=True)
        if vec1_angle < 0:
            vec1_angle += 360
        vec2 = self.np.array(C) - self.np.array(B)
        vec2_angle = self.np.angle(vec2[0] + vec2[1]*1j, deg=True)
        if vec2_angle < 0:
            vec2_angle += 360

        delta_angle = vec2_angle-vec1_angle
        if delta_angle < 0:
            delta_angle += 360

        return delta_angle

    #get turning angles between links
    def get_angles(self, linkage):
        coords = self.get_coords2(linkage)
        exp_coords = 2*coords
        angles = [0,0,0,0]
        for i in range(len(coords)):
            angle = self.calc_angle(exp_coords[i-1],
                  exp_coords[i], exp_coords[i+1])
            angles[i] = angle
        return tuple(angles)

    #get non-reflex angle between links
    def get_angles2(self, linkage):
        coords = self.get_coords2(linkage)
        exp_coords = 2*coords
        angles = [0,0,0,0]
        for i in range(len(coords)):
            angle = self.calc_angle(exp_coords[i-1],
                  exp_coords[i], exp_coords[i+1])
            angles[i] = 180-angle if angle <= 180 else -180+angle
        return tuple(angles)

    #determines turning angle is left or right
    def l_r(self, angle):
        if angle >= 180:
            return 'r'
        else:
            return 'l'

    #takes 3 coordinates and checks if its a left turn
    def isLeft(self, A, B, C):
        delta_angle = self.calc_angle(A, B, C)
        if delta_angle >= 0 and delta_angle <180:
            return True
        return False

    #takes at lease 3 coordinates in array form. counts left turns.
    def left_turns(self, args):
        steps = len(args)
        if steps<=2:
            raise ValueError

        exp_args = 2*args #expand arg list
        lefts = 0
        for i in range(steps):
            if self.isLeft(exp_args[i], exp_args[i+1], exp_args[i+2]):
                lefts += 1
            else:
                pass
        return lefts

    def is_typ_valid(self, linkage, value):
        lefts1 = self.left_turns(self.get_coords(linkage,0))
        lefts2 = self.left_turns(self.get_coords(linkage,1))
        if value=='convex':
            return True
        elif value=='concave':
            if (lefts1==1 or lefts1==3) or (lefts2==1 or lefts2==3):
                return True
        elif value=='crossing':
            if lefts1==2 or lefts2==2:
                return True
        return False

    #get possible linkage types based on link lengths
    def get_possible_types(self, linkage):
        lefts1 = self.left_turns(self.get_coords(linkage,0))
        lefts2 = self.left_turns(self.get_coords(linkage,1))
        if (lefts1==1 or lefts1==3) or (lefts2==1 or lefts2==3):
            return ('convex', 'concave')
        return ('convex', 'crossing')

    #turns crank over specified angle
    def turn_crank(self, linkage, angle):
        prev = self.np.array(self.get_coords2(linkage)[2])

        try:
            linkage.set_state(c_angle=linkage.crank_angle+angle,
                              typ='convex')
        except:
            raise ValueError
        else:
            pos_types = self.get_possible_types(linkage)

            linkage.set_state(typ=pos_types[0])
            C1 = self.np.array(self.get_coords2(linkage)[2])
            linkage.set_state(typ=pos_types[1])
            C2 = self.np.array(self.get_coords2(linkage)[2])

            d1 = self.np.linalg.norm(C1-prev)
            d2 = self.np.linalg.norm(C2-prev)

            if d1 <= d2:
                linkage.set_state(typ=pos_types[0])
            else:
                linkage.set_state(typ=pos_types[1])
                
    #turns crank over specified angle, 
    def turn_crank2(self, linkage, angle):
        cplr_fol_angle = self.get_angles(linkage)[2]
        l_r = self.l_r(cplr_fol_angle)

        try:
            linkage.set_state(c_angle=linkage.crank_angle+angle,
                              typ='convex')
        except:
            raise ValueError
        else:
            new_angle = self.get_angles(linkage)[2]
            new_l_r = self.l_r(new_angle)
            toggle = (abs(cplr_fol_angle)<0.01) or \
                     (abs(180-cplr_fol_angle)<0.01)
            if new_l_r != l_r and not toggle: #if toggle, favor convex
                pos_types = self.get_possible_types(linkage)
                linkage.set_state(typ=pos_types[1])

    #get instant center of coupler
    def get_inst_ctr(self, linkage):
        cr_vec = self.np.array(self.get_coords2(linkage)[1]) - \
                 self.np.array(self.get_coords2(linkage)[0])
        fl_vec = self.np.array(self.get_coords2(linkage)[2]) - \
                 self.np.array(self.get_coords2(linkage)[3])
        gr_vec = self.np.array(self.get_coords2(linkage)[3]) - \
                 self.np.array(self.get_coords2(linkage)[0])
        x1, y1 = cr_vec[0], cr_vec[1]
        x2, y2 = fl_vec[0], fl_vec[1]
        xg, yg = gr_vec[0], gr_vec[1]

        k_1 = (xg - yg*x2/y2)/(x1 - x2*y1/y2) #simultaneous equations

        int_coord = self.np.array(self.get_coords2(linkage)[0]) + k_1*cr_vec

        return((int_coord[0], int_coord[1]))

    #get instant moment due to force acting on **coupler**
    def get_inst_moment1(self, linkage, force, location):
        loc_vec = self.np.array((location[0], location[1], 0))
        ic_vec = self.np.append(self.get_inst_ctr(linkage), 0)
        r_vec = loc_vec - ic_vec

        f_vec = self.np.append(force, 0)

        moment_vec = self.np.cross(r_vec, f_vec)
        return moment_vec[2]

    #get instant moment due to force acting on **crank**
    def get_inst_moment2(self, linkage, force, location):
        loc_vec = self.np.array((location[0], location[1], 0))
        origin_vec = self.np.append(self.get_coords2(linkage)[0], 0)
        r_vec = loc_vec - origin_vec
        f_vec1 = self.np.append(force, 0)
        m_vec = self.np.cross(r_vec, f_vec1)

        ic_cplr_vec = -self.np.array(self.get_inst_ctr(linkage)) + \
                      self.np.array(self.get_coords2(linkage)[1])
        crank_vec = self.np.array(self.get_coords2(linkage)[1]) - \
                    self.np.array(self.get_coords2(linkage)[0])
        ax, ay = crank_vec

        fx, fy = self.np.linalg.inv([[ax, ay],[-ay, ax]]) @ (0, m_vec[2])
        moment = self.np.cross(self.np.append(ic_cplr_vec, 0), (fx, fy, 0))

        return moment[2]

    #get instant moment due to force acting on **follower**
    def get_inst_moment3(self, linkage, force, location):
        loc_vec = self.np.array((location[0], location[1], 0))
        origin_vec = self.np.append(self.get_coords2(linkage)[3], 0)
        r_vec = loc_vec - origin_vec
        f_vec1 = self.np.append(force, 0)
        m_vec = self.np.cross(r_vec, f_vec1)

        ic_cplr_vec = -self.np.array(self.get_inst_ctr(linkage)) + \
                      self.np.array(self.get_coords2(linkage)[2])
        flwr_vec = self.np.array(self.get_coords2(linkage)[2]) - \
                    self.np.array(self.get_coords2(linkage)[3])
        ax, ay = flwr_vec

        fx, fy = self.np.linalg.inv([[ax, ay],[-ay, ax]]) @ (0, m_vec[2])
        moment = self.np.cross(self.np.append(ic_cplr_vec, 0), (fx, fy, 0))

        return moment[2]

    @staticmethod
    def cos_law(b, c, alpha):
        from numpy import cos, sqrt, pi
        alpha_rad = alpha/180*pi
        a_square = b**2 + c**2 - 2*b*c*cos(alpha_rad)
        return sqrt(a_square)

    def area_triangle(self, A, B, C):
        xa, ya, xb, yb, xc, yc = A[0], A[1], B[0], B[1], C[0], C[1]
        area = abs(xa*(yb-yc)+xb*(yc-ya)+xc*(ya-yb))
        return area

    def isPtInTriangle(self, A, B, C, P):
        outer_area = self.area_triangle(A, B, C)
        PAB = self.area_triangle(P, A, B)
        PBC = self.area_triangle(P, B, C)
        PAC = self.area_triangle(P, A, C)
        if abs(PAB+PBC+PAC-outer_area)/outer_area < 0.001:
            return True
        return False

    def get_csfn_code(self):
        a, b, g, h = self.crank, self.follower, self.ground, self.coupler
        code = ''

        if g+h-a-b >= 0: code += '+'
        else: code += '-'
        if b+g-a-h >= 0: code += '+'
        else: code += '-'
        if b+h-a-g >= 0: code += '+'
        else: code += '-'

        return code

    #gets classification of linkage (not taking into account crank angle)
    def get_csfn(self):
        csfn = (['-','-','+','grashof','crank','crank'],
                ['+','+','+','grashof','crank','rocker'],
                ['+','-','-','grashof','rocker','crank'],
                ['-','+','-','grashof','rocker','rocker'],
                ['-','-','-','non-grashof','0-rocker','0-rocker'],
                ['-','+','+','non-grashof','pi-rocker','pi-rocker'],
                ['+','-','+','non-grashof','pi-rocker','0-rocker'],
                ['+','+','-','non-grashof','0-rocker','pi-rocker'])

        def get_row(linkage, csfn):
            code = linkage.get_csfn_code()
            row_number = False
            for i in range(len(csfn)):
                if csfn[i][0]+csfn[i][1]+csfn[i][2] == code:
                    row_number = i
                    break
            return row_number

        row = get_row(self, csfn)
        return((csfn[row][3], csfn[row][4], csfn[row][5]))

    #returns inverted states of linkage
    def invert(self, linkage, steps=1, flip=False):
        from copy import deepcopy
        out = deepcopy(linkage)
        
        states = [0,0,0,0,90,0,linkage.typ,(0,0)]
        indices = [0,1,2,3] * 2
        new_indices = [0,0,0,0]

        #find new lengths
        lengths = 2 * (linkage.crank, linkage.coupler,
                       linkage.follower, linkage.ground)
        for i in range(4):
            states[i] = lengths[i+steps]
            new_indices[i] = indices[i+steps]

        #find new lengths if flip
        temp = states[0:4]
        new_indices = 2 * new_indices
        if flip:
            for i in range(4):
                states[i] = temp[-i]
                new_indices[i] = new_indices[-i]
        else:
            pass

        #find origin
        states[7] = (self.get_coords2(linkage)*2)\
        [new_indices[0]+1 if flip else new_indices[0]]

        #find crank angle and ground angle
        pos3 = (self.get_coords2(linkage)*2)\
        [new_indices[3]+1 if flip else new_indices[3]]
        pos0 = (self.get_coords2(linkage)*2)\
        [new_indices[0]+1 if flip else new_indices[0]]
        pos1 = (self.get_coords2(linkage)*2)\
        [new_indices[1]+1 if flip else new_indices[1]]

        c_vec = self.np.array(pos1) - self.np.array(pos0)
        g_vec = self.np.array(pos3) - self.np.array(pos0)

        states[4] = self.np.angle(c_vec[0] + c_vec[1]*1j, deg=True)
        states[5] = self.np.angle(g_vec[0] + g_vec[1]*1j, deg=True)

        out.set_state(states[0], states[1], states[2], states[3],
                      states[4], states[5], states[6], states[7],)
        print('inverted linkage:', tuple(states))
        return out
    
    #returns x and y coordinates of joints over full range of crank rotation
    def get_coord_array(self, linkage):
        orig_turn = 1
        turn = orig_turn
        start_angle = linkage.crank_angle
        x = self.np.array([[self.get_coords2(linkage)[0][0],
                            self.get_coords2(linkage)[1][0],
                            self.get_coords2(linkage)[2][0],
                            self.get_coords2(linkage)[3][0]]])
        y = self.np.array([[self.get_coords2(linkage)[0][1],
                            self.get_coords2(linkage)[1][1],
                            self.get_coords2(linkage)[2][1],
                            self.get_coords2(linkage)[3][1]]])
        while True:
            try:
                self.turn_crank2(linkage, turn)
            except:
                turn = -turn
            finally:
                x = self.np.append(x, 
                                   [[self.get_coords2(linkage)[0][0],
                                     self.get_coords2(linkage)[1][0],
                                     self.get_coords2(linkage)[2][0],
                                     self.get_coords2(linkage)[3][0]]],
                                   axis=0)
                y = self.np.append(y, 
                                   [[self.get_coords2(linkage)[0][1],
                                     self.get_coords2(linkage)[1][1],
                                     self.get_coords2(linkage)[2][1],
                                     self.get_coords2(linkage)[3][1]]],
                                   axis=0)
                #print('crank angle:', linkage.crank_angle)
            if turn==orig_turn and linkage.crank_angle==start_angle-turn:
                break
            
        return x, y
    
    #returns generator of x and y coordinates of joints
    def get_coord_gen(self, linkage):
        orig_turn = 1
        turn = orig_turn
        x = self.np.array([[self.get_coords2(linkage)[0][0],
                            self.get_coords2(linkage)[1][0],
                            self.get_coords2(linkage)[2][0],
                            self.get_coords2(linkage)[3][0]]])
        y = self.np.array([[self.get_coords2(linkage)[0][1],
                            self.get_coords2(linkage)[1][1],
                            self.get_coords2(linkage)[2][1],
                            self.get_coords2(linkage)[3][1]]])
                    
        while True:
            try:
                self.turn_crank2(linkage, turn)
            except:
                turn = -turn
            finally:
                x = self.np.array([[self.get_coords2(linkage)[0][0],
                                    self.get_coords2(linkage)[1][0],
                                    self.get_coords2(linkage)[2][0],
                                    self.get_coords2(linkage)[3][0]]])
                y = self.np.array([[self.get_coords2(linkage)[0][1],
                                    self.get_coords2(linkage)[1][1],
                                    self.get_coords2(linkage)[2][1],
                                    self.get_coords2(linkage)[3][1]]])
            yield x, y