import copy
import sys
import numpy as np
import random
from Get_Problem import Get_Problem


class MonteCarlo:
    def __init__(self, Matrix, Job_Num, Machine_Num):
        self.Matrix = Matrix
        self.Job_Num = Job_Num
        self.Machine_Num = Machine_Num
        self.process = [len(Matrix[i]) for i in range(Job_Num)]
        self.total_process = sum(self.process)

    # Deconding
    def Decode(self, Chrome):
        OS = np.array(Chrome[:self.total_process], dtype=int)
        MS = np.array(Chrome[self.total_process: self.total_process * 2], dtype=int)
        T = []
        JM = []
        Start_End_time = []
        Site = 0
        Machine_list = []
        for i in self.process:
            JM.append(MS[Site: Site + i])
            Site += i
        for i in range(self.Job_Num):
            T_i = []
            Start_End_time_i = []
            for j in range(self.process[i]):
                O_j = self.Matrix[i][j]
                T_i.append(O_j[JM[i][j] - 1])
                Start_End_time_i.append([0, 0])
            T.append(T_i)
            Start_End_time.append(Start_End_time_i)
        Machine_Idle_Num = []
        Machine_Idle_range = []
        Machine_Idle_len = []
        for i in range(self.Machine_Num):
            Machine_Idle_Num.append(1)
            Machine_Idle_range.append([[0, 9999]])
            Machine_Idle_len.append([9999])
        Current_op_num = np.zeros(self.Job_Num, dtype=int)
        Makespan = 0
        for i in OS:
            Job = i - 1
            Operation = Current_op_num[Job]
            Machine = JM[Job][Operation] - 1
            Operation_time = T[Job][Operation]
            if Operation == 0:
                index = np.where(np.array(Machine_Idle_len[Machine]) >= Operation_time)[0][0]
                a = Machine_Idle_range[Machine][index][0]
                b = Machine_Idle_range[Machine][index][1]
                Start_End_time[Job][Operation][0] = a
                Start_End_time[Job][Operation][1] = a + Operation_time
                Machine_list.append([Job, Operation, Machine, Operation_time, a])
                if a + Operation_time == b:
                    Machine_Idle_range[Machine].remove(Machine_Idle_range[Machine][index])
                    Machine_Idle_len[Machine].remove(Machine_Idle_len[Machine][index])
                    Machine_Idle_Num[Machine] -= 1
                else:
                    Machine_Idle_range[Machine][index][0] = a + Operation_time
                    Machine_Idle_len[Machine][index] = b - a - Operation_time
            else:
                Precedent_Operation_End_time = Start_End_time[Job][Operation - 1][1]
                index = np.where(np.array(Machine_Idle_len[Machine]) >= Operation_time)[0]
                for j in index:
                    if Machine_Idle_range[Machine][j][1] >= Precedent_Operation_End_time + Operation_time:

                        a = Machine_Idle_range[Machine][j][0]
                        b = Machine_Idle_range[Machine][j][1]
                        c = Precedent_Operation_End_time + Operation_time
                        if a >= Precedent_Operation_End_time:
                            Start_End_time[Job][Operation][0] = a
                            Start_End_time[Job][Operation][1] = a + Operation_time
                            Machine_list.append([Job, Operation, Machine, Operation_time, a])
                            if b == a + Operation_time:
                                Machine_Idle_range[Machine].remove(
                                    Machine_Idle_range[Machine][j])
                                Machine_Idle_len[Machine].remove(
                                    Machine_Idle_len[Machine][j])
                                Machine_Idle_Num[Machine] -= 1
                            else:
                                Machine_Idle_range[Machine][j][0] = a + Operation_time
                                Machine_Idle_len[Machine][j] = b - a - Operation_time
                        else:
                            Start_End_time[Job][Operation][0] = Precedent_Operation_End_time
                            Start_End_time[Job][Operation][1] = c
                            Machine_list.append([Job, Operation, Machine, Operation_time, Precedent_Operation_End_time])
                            Machine_Idle_range[Machine][j][1] = Precedent_Operation_End_time
                            Machine_Idle_len[Machine][j] = Precedent_Operation_End_time - a
                            if b > c:
                                Machine_Idle_range[Machine].insert(j + 1, [c, b])
                                Machine_Idle_len[Machine].insert(j + 1, b - c)
                                Machine_Idle_Num[Machine] += 1
                        break
            if Start_End_time[Job][Operation][1] > Makespan:
                Makespan = Start_End_time[Job][Operation][1]
            Current_op_num[Job] += 1
            Machine_list1 = sorted(Machine_list, key=lambda x: x[4])
        return Makespan, Start_End_time, JM, T, Machine_list1

    #   The latest processing time calculation
    def Decode_Latest(self, Chrome, machine_list):
        Machine_list = copy.deepcopy(machine_list)
        job_time = np.zeros(self.Job_Num)
        machine_time = np.zeros(self.Machine_Num)
        machine_able = np.zeros(self.Machine_Num)
        job_able = np.zeros(self.Job_Num)
        Start_End_time = []
        Site = 0
        ope = 0
        k = 0
        T = []
        JM = []
        MS = np.array(Chrome[self.total_process: self.total_process * 2], dtype=int)
        for i in self.process:
            JM.append(MS[Site: Site + i])
            Site += i
        for i in range(self.Job_Num):
            T_i = []
            Start_End_time_i = []
            for j in range(self.process[i]):
                O_j = self.Matrix[i][j]
                T_i.append(O_j[JM[i][j] - 1])
                Start_End_time_i.append([0, 0])
            T.append(T_i)
            Start_End_time.append(Start_End_time_i)
        makespan = 0
        while ope < self.total_process:
            job = int(Machine_list[k][0])
            Ope = int(Machine_list[k][1])
            mac = int(Machine_list[k][2])
            tim = T[job][Ope]
            if machine_able[mac] == 0:
                if Ope > job_able[job]:
                    k = k + 1
                    machine_able[mac] = 1
                    continue
                else:
                    open_time = int(max(machine_time[mac], job_time[job]))
                    end_time = int(open_time + tim)
                    Start_End_time[job][Ope][:] = [open_time, end_time]
                    if end_time > makespan:
                        makespan = end_time
                    job_able[job] += 1
                    machine_time[mac] = end_time
                    job_time[job] = end_time
                    Machine_list = np.delete(Machine_list, k, axis=0)
                    k = 0
                    ope += 1
                    continue
            else:
                k = k + 1
                continue
        return makespan, Start_End_time


#  Calculation for RM
#  The following 4 parameters need to be modified by user, including 1 path, 1 particle code, 1 remanufacturing jobs definition, 1 instance type.
#   file path: the root directory of the target file
Processing_time, Job_Num, Machine_Num = Get_Problem(
    r'C:\Users\liuj9\Desktop\FJSSPinstances\1_Brandimarte\BrandimarteMk3.fjs')
gbestpop = np.array(
    [5, 4, 10, 7, 2, 11, 8, 4, 8, 8, 6, 3, 6, 15, 6, 15, 8, 4, 5, 13, 5, 2, 12, 9, 4, 13, 1, 15, 9, 11, 13, 1, 7, 4, 14,
     7, 1, 10, 2, 10, 10, 12, 3, 12, 15, 11, 15, 9, 14, 7, 7, 4, 14, 3, 14, 14, 7, 13, 3, 11, 3, 8, 11, 12, 6, 13, 1, 3,
     6, 14, 15, 12, 5, 15, 8, 10, 2, 13, 15, 2, 5, 11, 1, 10, 4, 7, 3, 12, 9, 6, 14, 6, 13, 11, 14, 2, 8, 1, 7, 10, 12,
     9, 15, 10, 15, 4, 14, 13, 3, 2, 5, 2, 2, 9, 11, 13, 11, 12, 11, 1, 4, 13, 1, 12, 4, 8, 5, 9, 9, 5, 10, 7, 5, 12, 9,
     8, 1, 7, 9, 2, 6, 6, 10, 8, 3, 3, 1, 5, 14, 6, 4, 3, 3, 2, 5, 5, 5, 1, 8, 6, 7, 1, 2, 5, 3, 7, 4, 7, 8, 6, 3, 2, 4,
     5, 8, 6, 4, 4, 1, 2, 4, 5, 3, 4, 6, 2, 5, 6, 1, 6, 7, 6, 5, 8, 4, 7, 5, 6, 3, 8, 2, 7, 4, 7, 3, 8, 4, 4, 5, 3, 8,
     6, 2, 1, 7, 2, 5, 2, 4, 6, 6, 1, 4, 7, 8, 8, 3, 2, 4, 3, 1, 3, 7, 6, 7, 8, 5, 2, 3, 4, 1, 7, 4, 4, 8, 4, 5, 6, 7,
     8, 2, 6, 8, 4, 6, 1, 3, 5, 8, 6, 8, 5, 7, 1, 2, 8, 5, 3, 8, 2, 8, 5, 4, 2, 6, 7, 4, 6, 5, 8, 7, 5, 2, 2, 6, 4, 6,
     5, 6, 1, 2, 6, 7, 4, 7, 1, 3, 5, 4, 2]
    )  # particle code. The following here is an encoding example for Mk03
Remanuf_Job = [1, 3,
               5]  # remanufacturing jobs definition. The following here [1,3,5] represents J2,J4,J6 are defined as remanufacturing jobs.
Benckmark_type = 1  # instance type，1 for Brandimarte， 2 for Fattahi， 3 for Kacem

Makespan, Start_End_time1, JM, _, m_list = MonteCarlo(Processing_time, Job_Num, Machine_Num).Decode(gbestpop)
sum_makespan = 0
for k in range(1000):  # 1000 Monte Carlo simulation runs
    Processing_time1 = copy.deepcopy(Processing_time)
    for v in Remanuf_Job:
        for i in range(len(Processing_time1[v])):
            for j in range(len(Processing_time1[v][i])):
                if Processing_time1[v][i][j] != 9999:
                    if Benckmark_type == 1:  # if Brandimarte instance, perform [pt,pt+5] perturbation setting
                        Processing_time1[v][i][j] = Processing_time1[v][i][j] + random.randint(0, 5)
                    elif Benckmark_type == 2:  # if Fattahi instance, perform [pt,1.3pt] perturbation setting
                        Processing_time1[v][i][j] = (1 + 0.01 * random.randint(0, 30)) * Processing_time1[v][i][j]
                    else:  # if Kacem instance, perform [pt,1.5pt] perturbation setting
                        Processing_time1[v][i][j] = (1 + 0.01 * random.randint(0, 50)) * Processing_time1[v][i][j]
    makespan, Start_End_time2 = MonteCarlo(Processing_time1, Job_Num, Machine_Num).Decode_Latest(gbestpop, m_list)
    sum_makespan = sum_makespan + makespan
RM = (sum_makespan / 1000 - Makespan) / Makespan  # RM takes the average of 1000 Monte Carlo simulation runs
print('RM:', RM)
