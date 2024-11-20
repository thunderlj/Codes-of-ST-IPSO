import os


# Conversion of the Brandimarte/Fattahi/Kacem dataset (.txt) into matrix format
def Get_Problem(path):
    if os.path.exists(path):
        with open(path, 'r') as data:
            List = data.readlines()
            P = []
            for line in List:
                list_ = [int(number) for number in line.split()]
                P.append(list_)
            for k in P:
                if k == []:
                    P.remove(k)
            J_number = P[0][0]
            M_number = P[0][1]
            Problem = []
            for J_num in range(1, len(P)):
                O_num = P[J_num][0]
                for Oij in range(O_num):
                    O_j = []
                    next = 1
                    while next < len(P[J_num]):
                        M_Oj = [0 for Oji in range(M_number)]
                        M_able = P[J_num][next]
                        able_set = P[J_num][next + 1:next + 1 + M_able * 2]
                        next = next + 1 + M_able * 2
                        for i_able in range(0, len(able_set), 2):
                            M_Oj[able_set[i_able] - 1] = able_set[i_able + 1]
                        O_j.append(M_Oj)
                Problem.append(O_j)
            for i_1 in range(len(Problem)):
                for j_1 in range(len(Problem[i_1])):
                    for k_1 in range(len(Problem[i_1][j_1])):
                        if Problem[i_1][j_1][k_1] == 0:
                            Problem[i_1][j_1][k_1] = 9999
    else:
        print('Something is wrong with the path')
    return Problem, J_number, M_number
