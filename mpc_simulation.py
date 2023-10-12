#!/usr/bin/env python
# -*- coding: utf-8 -*-
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import math

class MPCController():

    def __init__(self):
        self.T = 0.1
        self.N = 7  # division number
        self.l_roll = 2.3
        self.m_roll = 14.5
        self.l_pitch = 3.8
        self.m_pitch = 10.5
        self.l_height = 0.5
        self.m_height = 26.5
        self.l_depth = 0.3
        self.m_depth = 23.5
        self.q_height = 30.0

        self.u_1s = np.ones(self.N) *0.1 #surge
        self.u_2s = np.ones(self.N) *0.1 #heave
        self.u_3s = np.ones(self.N) *0.1 #yaw

        self.dummy_u_1s = np.ones(self.N) *0.1
        self.dummy_u_2s = np.ones(self.N) *0.1
        self.dummy_u_3s = np.ones(self.N) *0.1

        self.history_u_1 = []
        self.history_u_2 = []
        self.history_u_3 = []
        self.history_dummy_u_1 = []
        self.history_dummy_u_2 = []
        self.history_dummy_u_3 = []
        self.history_raw_1 = []
        self.history_raw_2 = []
        self.history_raw_3 = []
        self.history_f = []
        self.Q = np.diag([7.0, 0.0, 17]) 
        self.Q_vel = np.diag([3.5, 0.0, 5.5]) 
        self.calc_input(1,1,1,1,1,1,1,1,1,1)

    def calc_input(self, roll, pitch, height, depth, vel_roll, vel_pitch, vel_height, vel_depth, state_ref, time_1):
        #system states
        ubvms_vel_x = ca.SX.sym('ubvms_vel_x')
        ubvms_vel_y = ca.SX.sym('ubvms_vel_y')
        ubvms_roll = ca.SX.sym('ubvms_roll')
        ubvms_vel_roll = ca.SX.sym('ubvms_vel_roll')
        ubvms_pitch = ca.SX.sym('ubvms_pitch')
        ubvms_vel_pitch = ca.SX.sym('ubvms_vel_pitch')
        ubvms_z = ca.SX.sym('ubvms_z')
        ubvms_vel_z = ca.SX.sym('ubvms_vel_z')
        states_vel = ca.vertcat(*[ubvms_vel_roll, ubvms_vel_pitch, ubvms_vel_z])
        n_states = states_vel.size()[0] #（n_states, 1）
        
        thruster_1 = ca.SX.sym('thruster_1')
        thruster_2 = ca.SX.sym('thruster_2')
        thruster_3 = ca.SX.sym('thruster_3')
        thruster_4 = ca.SX.sym('thruster_4')
        controls = ca.vertcat(thruster_1, thruster_2, thruster_3, thruster_4) 
        n_controls = controls.size()[0] 
        #model acc
        rhs_acc = ca.vertcat(*[0, 0, thruster_1*self.l_roll/self.m_roll+thruster_2*self.l_roll/self.m_roll-thruster_3*self.l_roll/self.m_roll
                         -thruster_4*self.l_roll/self.m_roll, \
                         thruster_1*self.l_pitch/self.m_pitch-thruster_2*self.l_pitch/self.m_pitch-thruster_3*self.l_pitch/self.m_pitch
                         +thruster_4*self.l_pitch/self.m_pitch-(25*np.sin(ubvms_pitch)+5*ubvms_vel_pitch)*self.l_pitch/self.m_pitch, \
                        self.l_height/self.m_height*(thruster_1+thruster_2+thruster_3+thruster_4)])

        f_acc = ca.Function('f_acc', [controls, ubvms_pitch,ubvms_vel_pitch], [rhs_acc], ['control_input','ubvms_pitch','ubvms_vel_pitch'], ['rhs_acc'])
        U = ca.SX.sym('U', n_controls, self.N)
        X_vel = ca.SX.sym('X_vel', n_states+2, self.N+1)
        X = ca.SX.sym('X', n_states, self.N+1)
        X_vel_ref = ca.SX.sym('X_vel_ref', n_states+2)
        X_ref = ca.SX.sym('X_ref', 5+3*3)
        U_ref = ca.SX.sym('U_ref', n_controls)

        states = ca.vertcat(*[ubvms_roll, ubvms_pitch, ubvms_z])
        states_vel_1 = ca.vertcat(*[ubvms_vel_x, ubvms_vel_y, ubvms_vel_roll, ubvms_vel_pitch, ubvms_vel_z])
        rhs = ca.vertcat(*[ubvms_vel_roll, ubvms_vel_pitch, -np.sin(ubvms_pitch)*ubvms_vel_x \
                           + np.sin(ubvms_roll)*np.cos(ubvms_pitch)*ubvms_vel_y + np.cos(ubvms_roll)*np.cos(ubvms_pitch)*ubvms_vel_z])
        f_vel = ca.Function('f_vel', [states, states_vel_1], [rhs], ['input_state', 'control_input'], ['rhs'])
        X_vel[:, 0] = X_ref[:5]
        X[:, 0] = X_ref[5:8]
        for i in range(self.N):
            f_acc_value = f_acc(U[:, i], X[1, i],X_vel[3, i])
            X_vel[:, i+1] = X_vel[:, i] + f_acc_value*self.T
            f_vel_value = f_vel(X[:, i], X_vel[:, i])
            X[:, i+1] = X[:, i] + f_vel_value*self.T
        ff = ca.Function('ff', [U, X_ref], [X_vel, X], ['input_U', 'target_state'], ['vel', 'horizon_states'])

        x_vel_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        x_0 = np.array([0.0, 0.3, 0.0]).reshape(-1, 1)
        desired_x = np.array([0.0, 0.0, 2.0]).reshape(-1, 1)
        desire_vel_x = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
        u0 = np.array([0.0, 0.0, 0.0, 0.0] * self.N).reshape(-1, 4)
        x_c = []
        u_c = []
        mpciter = 0
        start_time = time.time()
        iter_dis=0

        #while mpciter<200:
        while (np.linalg.norm(x_0[2]-desired_x[2])>1e-2) or \
                (np.linalg.norm(x_vel_0[4]-desire_vel_x[2])>1e-2) or \
                (np.linalg.norm(x_0[1]-desired_x[1])>1e-2) or \
                (np.linalg.norm(x_vel_0[3]-desire_vel_x[1])>1e-2) or mpciter<150:
            if mpciter == 90:
                x_0[2]=1.2
                x_0[1]=-0.2
            R = np.diag([0.001, 0.001, 0.001, 0.001])
            R_change = np.diag([0.01, 0.01, 0.01, 0.01])
            obj = 0
            for i in range(self.N):
                X_state_errors = X[:, i] - X_ref[8:11]
                X_vel_state_errors = X_vel[2:5, i] - X_ref[11:14]
                if i < 6:
                    U_change = U[:, i+1] - U[:, i]
                else:
                    U_change = np.array([0, 0, 0, 0]).reshape(-1,1)
                obj = obj + ca.mtimes([X_state_errors.T, self.Q, X_state_errors]) \
                      + ca.mtimes([X_vel_state_errors.T, self.Q_vel, X_vel_state_errors]) \
                      + ca.mtimes([U[:, i].T, R, U[:, i]]) \
                      + ca.mtimes([U_change.T, R_change, U_change])
                #print(i,U_change)
            g = []
            for i in range(self.N+1):
                for j in range(5):
                    g.append(X_vel[j, i])
                for k in range(2):
                    g.append(X[k, i])
            nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': X_ref, 'g': ca.vertcat(*g)}
            opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
            solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
            lbg = [-0.3, -0.15, -0.15, -0.15, -0.5, -np.pi/2, -np.pi/2]*(self.N+1)#without z
            ubg = [0.3, 0.15, 0.15, 0.15, 0.5, np.pi/2, np.pi/2]*(self.N+1)
            #lbg = -0.2
            #ubg = 0.6
            lbx = []
            ubx = []
            for _ in range(self.N):
                for _ in range (n_controls):
                    lbx.append(-25)
                    ubx.append(25)

            c_p = np.concatenate((x_vel_0, x_0, desired_x, desire_vel_x))
            init_control = ca.reshape(u0, -1, 1)
            res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
            u_sol = ca.reshape(res['x'], n_controls, self.N) 
            f_value = ff(u_sol, c_p)
            x_c.append(f_value)
            x_vel_0 = f_value[0][:,1]
            x_0 = f_value[1][:,1]
            mpciter += 1
            mpc_time = mpciter*self.T
            q_height = 35.0/(1 + np.exp(-3*abs(-desired_x[2]+x_0[2])))
            q_vel_height = 10.5/(np.exp(3*abs(-desired_x[2]+x_0[2])))
            q_vel_pitch = 2.5/(np.exp(3*abs(-desired_x[1]+x_0[1])))
            print(q_vel_pitch,mpciter)
            q_pitch = 14.0/(1 + np.exp(-3*abs(-desired_x[1]+x_0[1])))
            self.Q = np.diag([7.0, q_pitch, q_height])
            self.Q_vel = np.diag([3.5, q_vel_pitch, q_vel_height]) 
            #print(u_sol[:, 0])
            u_c.append(u_sol[:, 0])
            #save data
            with open('Mpc_data.txt', 'a') as DataFileRec:
                DataFileRec.write(str('%.2f' % mpc_time) + ' ')
                DataFileRec.write(str('%.2f' % f_value[0][0, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[0][1, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[0][2, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[0][3, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[0][4, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[1][0, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[1][1, 1]) + ' ')
                DataFileRec.write(str('%.2f' % f_value[1][2, 1]) + ' ')
                DataFileRec.write(str('%.2f' % u_sol[0, 0]) + ' ')
                DataFileRec.write(str('%.2f' % u_sol[1, 0]) + ' ')
                DataFileRec.write(str('%.2f' % u_sol[2, 0]) + ' ')
                DataFileRec.write(str('%.2f' % u_sol[3, 0]) + ' ')
                DataFileRec.write(str('%.2f' % desired_x[2]) + '\n')
        print(time.time()-start_time)
        print(mpciter)
        u_plt = np.zeros(len(u_c)*4).reshape(-1,4)
        x_vel_plt = np.zeros(len(x_c)*5).reshape(-1,5)
        x_plt = np.zeros(len(x_c)*3).reshape(-1,3)
        fig = plt.figure()
        for j in range(4):
            for i in range(len(u_c)):
                u_plt[i,j]=u_c[i][j]
        for i in range(len(x_c)):
            for k in range(5):
                x_vel_plt[i,k]=x_c[i][0][k,1]
        for i in range(len(x_c)):
            for k in range(3):
                x_plt[i,k]=x_c[i][1][k,1]
        plt.plot(x_vel_plt)
        plt.plot(x_plt)
        plt.show()


if __name__ == "__main__":
    try:
        MPCController()
    except KeyboardInterrupt:
        print("Shutting down interacter node")