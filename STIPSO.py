import copy
import os
import sys
import numpy as np
import random
from Get_Problem import Get_Problem

class ST_IPSO:
    def __init__(self, Matrix, Job_Num, Machine_Num, Pop_size, MB_size, Max_Iter, Tabulist_Length):
        self.Matrix = Matrix                                                         # Processing_time matrix
        self.Pop_size = Pop_size                                                     # Population Size
        self.Job_Num = Job_Num                                                       # total number of jobs corresponding to a instance
        self.Machine_Num = Machine_Num                                               # total number of machiens corresponding to a instance
        self.MB_size = MB_size                                                       # Memory bank size
        self.Max_Iter = Max_Iter                                                     # Maximum number of iterations
        self.Tabulist_Length = Tabulist_Length                                       # Length of tabu list
        self.process= [len( Matrix[i]) for i in range(Job_Num)]                      # process number of each job
        self.total_process = sum(self.process)                                       # total number of processes for all jobs

    # Population Initialization，70% follows random rule and the remain 30% follows SPT
    def Initialization(self):
        Population = np.zeros((self.Pop_size, self.total_process * 2))
        fitness_value = np.zeros(self.Pop_size)
        for i in range(int(self.Pop_size)):
            OS = []
            for j in range(self.Job_Num):
                OS = np.concatenate((OS, (j + 1) * np.ones(self.process[j])))
            Population[i][:self.total_process] = OS
            np.random.shuffle(Population[i][:self.total_process])
            MS = []
            for j in range(self.Job_Num):
                for k in range(self.process[j]):
                    if i < 0.7 * self.Pop_size:
                        aviable_machine_index = np.where(np.array(self.Matrix[j][k]) != 9999)
                        MS = np.concatenate((MS, [np.random.choice(aviable_machine_index[0]) + 1]))
                    else:
                        machine_SPT_index = np.where(
                            np.array(self.Matrix[j][k]) == np.min(np.array(self.Matrix[j][k])))
                        MS = np.concatenate((MS, [np.random.choice(machine_SPT_index[0]) + 1]))
            Population[i][self.total_process: self.total_process * 2] = MS
            fitness_value[i] = self.Decode(Population[i])[0]
        # Population Initialization in Memory bank
        arr1 = copy.deepcopy(Population)
        arr2 = copy.deepcopy(fitness_value)
        arrayIndex = arr2.argsort()
        arr1 = arr1[arrayIndex]
        arr2 = arr2[arrayIndex]
        MB_Population = arr1[0 : self.MB_size]
        MB_fitnessvalue = arr2[0 : self.MB_size]
        return Population, fitness_value, MB_Population, MB_fitnessvalue

    # Particle Decoding
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
                Machine_list.append([Job, Operation, Machine, Operation_time, a + Operation_time])
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
                            Machine_list.append([Job, Operation, Machine, Operation_time, a + Operation_time])
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
                            Machine_list.append([Job, Operation, Machine, Operation_time, c])
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

    # The latest processing time calculation
    def Decode_Latest(self, Chrome, machine_list):
        machine_list.reverse()
        job_time=np.zeros(self.Job_Num)
        machine_time=np.zeros(self.Machine_Num)
        machine_able=np.zeros(self.Machine_Num)
        job_able = copy.deepcopy(self.process)
        Start_End_time = []
        Site = 0
        ope=0
        k = 0
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
            Start_End_time.append(Start_End_time_i)
        while ope <  self.total_process:
            job=int(machine_list[k][0])
            Ope=int(machine_list[k][1])
            mac=int(machine_list[k][2])
            tim=int(machine_list[k][3])
            if machine_able[mac]==0:
                if Ope +1 < job_able[job]:
                    k = k + 1
                    machine_able[mac] = 1
                    continue
                else:
                    open_time=max(machine_time[mac],job_time[job])
                    end_time=open_time+tim
                    Start_End_time[job][Ope][:]=[open_time,end_time]
                    job_able[job]=job_able[job]-1
                    machine_able=np.zeros(self.Machine_Num)
                    machine_time[mac]=end_time
                    job_time[job]=end_time
                    machine_list=np.delete(machine_list,k,axis=0)
                    k=0
                    ope = ope+1
                    continue
            else:
                k=k+1
                continue
        return job_time, Start_End_time,

    # Calculate the slack time, Total_slack represents the total slack for different jobs.
    def Total_slack(self, Chrome):
        Makespan, Start_End_time1, _, _,m_list = self.Decode(Chrome)
        Start_End_time2 = self.Decode_Latest(Chrome,m_list)[1]
        Total_Slack = []
        for i in range(self.Job_Num):
            total_slack = 0
            for j in range(self.process[i]):
                total_slack = total_slack + Makespan - Start_End_time1[i][j][0] - Start_End_time2[i][j][1]
            Total_Slack.append(total_slack)
        return Total_Slack

    #   Crossover Operations for OS segment
    def POX_cross(self, chrome1, chrome2):
        process_parent1 = copy.deepcopy(chrome1[0:self.total_process])
        process_parent2 = copy.deepcopy(chrome2[0:self.total_process])
        machine_parent1 = copy.deepcopy(chrome1[self.total_process: self.total_process * 2])
        machine_parent2 = copy.deepcopy(chrome2[self.total_process: self.total_process * 2])
        Job_list = list(range(1, self.Job_Num + 1))
        random.shuffle(Job_list)
        r = random.randint(1, self.Job_Num - 1)
        set1 = Job_list[0: r]
        set2 = Job_list[r: self.Job_Num]
        process_child1 = list(np.zeros(self.total_process))
        for i, Job in enumerate(process_parent1):
            if Job in set1:
                process_child1[i] = Job
        for Job in process_parent2:
            if Job in set2:
                site = process_child1.index(0)
                process_child1[site] = Job
        New_chrome1 = np.array(process_child1 + list(machine_parent1))
        process_child2 = list(np.zeros(self.total_process))
        for i, Job in enumerate(process_parent2):
            if Job in set1:
                process_child2[i] = Job
        for Job in process_parent1:
            if Job in set2:
                site = process_child2.index(0)
                process_child2[site] = Job
        New_chrome2 = np.array(process_child2 + list(machine_parent2))
        return New_chrome1, New_chrome2

    #   Crossover Operations for MS segment
    def MPX_cross(self, Chrome1, Chrome2):
        process_parent1 = copy.deepcopy(Chrome1[0:self.total_process])
        process_parent2 = copy.deepcopy(Chrome2[0:self.total_process])
        machine_parent1 = copy.deepcopy(Chrome1[self.total_process:self.total_process * 2])
        machine_parent2 = copy.deepcopy(Chrome2[self.total_process:self.total_process * 2])
        machine_child1 = copy.deepcopy(machine_parent2)
        machine_child2 = copy.deepcopy(machine_parent1)
        r = np.random.randint(2, size=self.total_process, dtype=int)
        for i, Machine in enumerate(machine_parent1):
            if r[i] == 0:
                machine_child1[i] = Machine
        for i, Machine in enumerate(machine_parent2):
            if r[i] == 0:
                machine_child2[i] = Machine
        New_chrome1 = np.array(list(process_parent1) + list(machine_child1))
        New_chrome2 = np.array(list(process_parent2) + list(machine_child2))
        return New_chrome1, New_chrome2

    # Mutation Operator for MS segment
    def Mutation_machine(self, Chrome):
        New_Chrome = copy.deepcopy(Chrome)
        r = random.randint(0, self.total_process - 1)
        Current_Machine = New_Chrome[self.total_process + r]
        R = r + 1
        for i, value in enumerate(self.process):
            R = R - value
            if R <= 0:
                Job = i
                Operation_num = R + value - 1
                available_machine_index = list(np.where(np.array(self.Matrix[Job][Operation_num]) != 9999)[0] + 1)
                if len(available_machine_index) > 1:
                    available_machine_index.remove(Current_Machine)
                    Replace_Machine = np.random.choice(available_machine_index)
                else:
                    Replace_Machine = Current_Machine
                break
        New_Chrome[self.total_process + r] = Replace_Machine
        return  New_Chrome

    # Mutation Operator for OS segment
    def Mutation_process(self, Chrome):
        New_Chrome = copy.deepcopy(Chrome)
        numbers = list(range(0, self.total_process))
        random.shuffle(numbers)
        r1 = numbers[0]
        r2 = numbers[1]
        process1 = copy.deepcopy(New_Chrome[r1])
        process2 = copy.deepcopy(New_Chrome[r2])
        New_Chrome[r1] = process2
        New_Chrome[r2] = process1
        return New_Chrome

    # Tabu List Updating
    def update_Tabulist(self, Chrome, fitness, Tabu_list1, Tabu_list2):
        if len(Tabu_list1) < self.Tabulist_Length:
            Tabu_list1.append(Chrome.tolist())
            Tabu_list2.append(fitness)
        else:
            Tabu_list1.pop(0)
            Tabu_list2.pop(0)
            Tabu_list1.append(Chrome.tolist())
            Tabu_list2.append(fitness)
        return Tabu_list1, Tabu_list2

    # Tabu search based local searching strategy
    def Tabu_search(self, Chrome, STIPSO_Current_iter):
        Current_Chrome = copy.deepcopy(Chrome)
        Best_Chrome = copy.deepcopy(Current_Chrome)
        Best_fitness = self.Decode(Best_Chrome)[0]
        Tabu_list1 = []
        Tabu_list2 = []
        TS_Max_Inter = int(150 * STIPSO_Current_iter / self.Max_Iter)
        for i in range(TS_Max_Inter):
            arr1 = []
            arr2 = []
            # Neighbour structure: LS1
            New_Chrome1 = copy.deepcopy(Current_Chrome)
            numbers = list(range(0, self.total_process))
            random.shuffle(numbers)
            position1 = numbers[0]
            position2 = numbers[1]
            temp1 = New_Chrome1[position1]
            temp2 = New_Chrome1[position2]
            New_Chrome1[position1] = temp2
            New_Chrome1[position2] = temp1
            if New_Chrome1.tolist() in Tabu_list1:
                pass
            else:
                fitness_New1 = self.Decode(New_Chrome1)[0]
                self.update_Tabulist(New_Chrome1, fitness_New1, Tabu_list1, Tabu_list2)
                arr1.append(New_Chrome1)
                arr2.append(fitness_New1)
            # Neighbour structure: LS2
            list_Chrome = copy.deepcopy(Current_Chrome)
            spots = list(range(0, self.total_process))
            random.shuffle(spots)
            spot_1 = min(spots[0], spots[1])
            spot_2 = max(spots[0], spots[1])
            insert_process = list_Chrome[spot_2]
            list_Chrome = list(list_Chrome)
            list_Chrome.pop(spot_2)
            list_Chrome.insert(spot_1, insert_process)
            New_Chrome2 = np.array(list_Chrome)
            if New_Chrome2.tolist() in Tabu_list1:
                pass
            else:
                fitness_New2 = self.Decode(New_Chrome2)[0]
                self.update_Tabulist(New_Chrome2, fitness_New2, Tabu_list1, Tabu_list2)
                arr1.append(New_Chrome2)
                arr2.append(fitness_New2)
            # Neighbour structure: LS3
            New_Chrome3 = copy.deepcopy(Current_Chrome)
            start = random.randint(0, int(self.total_process / 2))
            end = random.randint(start, self.total_process - 1)
            while start < end:
                temp3 = New_Chrome3[start]
                temp4 = New_Chrome3[end]
                New_Chrome3[start] = temp4
                New_Chrome3[end] = temp3
                start = start + 1
                end = end - 1
            if New_Chrome3.tolist() in Tabu_list1:
                pass
            else:
                fitness_New3 = self.Decode(New_Chrome3)[0]
                self.update_Tabulist(New_Chrome3, fitness_New3, Tabu_list1, Tabu_list2)
                arr1.append(New_Chrome3)
                arr2.append(fitness_New3)
            # Neighbour structure: LS4
            New_Chrome4 = self.Mutation_machine(copy.deepcopy(Current_Chrome))
            if New_Chrome4.tolist() in Tabu_list1:
                pass
            else:
                fitness_New4 = self.Decode(New_Chrome4)[0]
                self.update_Tabulist(New_Chrome4, fitness_New4, Tabu_list1, Tabu_list2)
                arr1.append(New_Chrome4)
                arr2.append(fitness_New4)
            if len(arr1) > 0:
                New_Bestfitness = min(arr2)
                New_BestChrome = arr1[np.argmin(arr2)]
                Current_Chrome = copy.deepcopy(New_BestChrome)
                if New_Bestfitness < Best_fitness:
                    Best_Chrome = New_BestChrome
                    Best_fitness = New_Bestfitness
                    Tabu_list1.clear()
                    Tabu_list2.clear()
        return Best_Chrome, Best_fitness

    # main function
    def main(self):
        Population, fitness_value, MB_Population, MB_fitnessvalue = self.Initialization()
        globalBestPop, globalBestFitness = Population[fitness_value.argmin()].copy(), fitness_value.min()
        pbestpop, pbestfitness = Population.copy(), fitness_value.copy()
        Bestfitness_for_EachIter = np.zeros(self.Max_Iter)
        pso_base = np.zeros(self.Max_Iter)
        n = int(0.5 * self.Pop_size)
        for k in range(6):      # Loop count Ls: 6
            # The Stage 1
            for i in range(self.Max_Iter):
                X = np.array(copy.deepcopy(pbestfitness), dtype=int)
                count = 0
                for j in range(self.Pop_size):
                    if random.random() < (1 - i / self.Max_Iter):
                        if random.random() < 0.8:      #  Crossover Probability pc: 0.8
                            MB_index = random.randint(0, self.MB_size - 1)
                            p1, p2 = self.MPX_cross(Population[j], MB_Population[MB_index])
                            p3, p4 = self.POX_cross(p1, p2)
                            p5, p6 = self.MPX_cross(Population[j], globalBestPop)
                            p7, p8 = self.POX_cross(p5, p6)
                            f1, f2, f3, f4 = self.Decode(p3)[0],self.Decode(p4)[0],self.Decode(p7)[0],self.Decode(p8)[0]
                            arr1 = [fitness_value[j], f1, f2, f3, f4]
                            arr2 = [Population[j], p3, p4, p7, p8]
                            Population[j] = arr2[np.argmin(arr1)]
                            fitness_value[j] = min(arr1)
                        if random.random() < 0.1:     #   Mutatuin Probability Pm: 0.1
                            p1 = self.Mutation_process(Population[j])
                            p1 = self.Mutation_machine(p1)
                            f1 = self.Decode(p1)[0]
                            if fitness_value[j] > f1:
                                Population[j] = p1
                                fitness_value[j] = f1
                    else:
                        if fitness_value[j] <= np.partition(X, n)[n]:
                            if count <= n:
                                Population[j], fitness_value[j] = self.Tabu_search(Population[j], i)
                                count = count + 1
                    if fitness_value[j] < pbestfitness[j]:
                        pbestfitness[j] = fitness_value[j]
                        pbestpop[j] = copy.deepcopy(Population[j])
                    if fitness_value[j] < np.max(MB_fitnessvalue):
                        MB_fitnessvalue[np.argmax(MB_fitnessvalue)] = fitness_value[j]
                        MB_Population[np.argmax(MB_fitnessvalue)] = copy.deepcopy(Population[j])
                if pbestfitness.min() < globalBestFitness:
                    globalBestFitness = pbestfitness.min()
                    globalBestPop = copy.deepcopy(Population[pbestfitness.argmin()])
                print("Current Iteration Number is ：", i)
                # 每次迭代后种群适应度最小值的变化情况
                Bestfitness_for_EachIter[i] = fitness_value.min()
                print("When Iteration Number ", i, ", the minimum fitness value is ", Bestfitness_for_EachIter[i])
                # 全局最优解的变化情况
                pso_base[i] = globalBestFitness
                print("Global best solution is ", pso_base[i])
            # The Stage 2
            indices = [index for index, value in enumerate(fitness_value) if value == min(Bestfitness_for_EachIter)]
            length = len(indices)
            new_Population = [Population[i] for i in indices]  # new_Population is the optimal solution set with minimum makespan
            new_FitnessValue = np.zeros(length)
            Remanuf_Job = [1, 3, 5]  # remanufacturing jobs definition. The following here [1,3,5] represents J2,J4,J6 are defined as remanufacturing jobs.
            for i in range(length):
                Total_Slack = self.Total_slack(new_Population[i])
                new_FitnessValue[i] = sum(Total_Slack[j] for j in Remanuf_Job)
            arrIndex = new_FitnessValue.argsort()[::-1]
            new_Population = np.array(new_Population)[arrIndex]
            new_FitnessValue = new_FitnessValue[arrIndex]
            X = copy.deepcopy(new_Population[0])
            globalBestPop = copy.deepcopy(new_Population[0])
            new_m_FitnessValue = np.zeros(self.MB_size)
            for i in range(self.MB_size):
                Total_Slack = self.Total_slack(MB_Population[i])
                new_m_FitnessValue[i] = sum(Total_Slack[j] for j in Remanuf_Job)
            if length < self.MB_size:
                continue
            else:
                for j in range(length):
                    if new_FitnessValue[j] > np.min(new_m_FitnessValue):
                        MB_fitnessvalue[np.argmin(new_m_FitnessValue)] = self.Decode(new_Population[j])[0]
                        MB_Population[np.argmin(new_m_FitnessValue)] = copy.deepcopy(new_Population[j])
        print("The best solution yielded by ST_IPSO：", globalBestPop)
        print("Makepsan of the best solution yielded by ST_IPSO：", min(pso_base))
        print("TS value of the best solution yielded by ST_IPSO：", new_FitnessValue[0])
        return globalBestPop
if __name__ == "__main__":
    # The remanufacturing jobs definition needs modification by user for instances with different problem size！！！！ in line441！！！！
    Processing_time, Job_Num, Machine_Num = Get_Problem(r'C:\Users\liuj9\Desktop\FJSSPinstances\1_Brandimarte\BrandimarteMk3.fjs')  #   file path: the root directory of the target file. User modification!!!
    globalBestPop = ST_IPSO(Processing_time,Job_Num, Machine_Num, Pop_size=150, MB_size=15,
            Max_Iter=200,Tabulist_Length=10).main()