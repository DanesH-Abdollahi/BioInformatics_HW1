###################################### BioInformatics Spring 2022 ######################################
###################################### DanesH Abdollahi - 9731314 ######################################
######################################## Semi-Global Alignment #########################################

import numpy as np

PAM250 = {
    'A': {'A':  2, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N':  0, 'P':  1, 'Q':  0, 'R': -2, 'S':  1, 'T':  1, 'V':  0, 'W': -6, 'Y': -3},
    'C': {'A': -2, 'C': 12, 'D': -5, 'E': -5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4, 'P': -3, 'Q': -5, 'R': -4, 'S':  0, 'T': -2, 'V': -2, 'W': -8, 'Y':  0},
    'D': {'A':  0, 'C': -5, 'D':  4, 'E': 3, 'F': -6, 'G':  1, 'H':  1, 'I': -2, 'K':  0, 'L': -4, 'M': -3, 'N':  2, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
    'E': {'A':  0, 'C': -5, 'D':  3, 'E': 4, 'F': -5, 'G':  0, 'H':  1, 'I': -2, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
    'F': {'A': -3, 'C': -4, 'D': -6, 'E': -5, 'F':  9, 'G': -5, 'H': -2, 'I':  1, 'K': -5, 'L':  2, 'M':  0, 'N': -3, 'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W':  0, 'Y':  7},
    'G': {'A':  1, 'C': -3, 'D':  1, 'E': 0, 'F': -5, 'G':  5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P':  0, 'Q': -1, 'R': -3, 'S':  1, 'T':  0, 'V': -1, 'W': -7, 'Y': -5},
    'H': {'A': -1, 'C': -3, 'D':  1, 'E': 1, 'F': -2, 'G': -2, 'H':  6, 'I': -2, 'K':  0, 'L': -2, 'M': -2, 'N':  2, 'P':  0, 'Q':  3, 'R':  2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y':  0},
    'I': {'A': -1, 'C': -2, 'D': -2, 'E': -2, 'F':  1, 'G': -3, 'H': -2, 'I':  5, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -5, 'Y': -1},
    'K': {'A': -1, 'C': -5, 'D':  0, 'E': 0, 'F': -5, 'G': -2, 'H':  0, 'I': -2, 'K':  5, 'L': -3, 'M':  0, 'N':  1, 'P': -1, 'Q':  1, 'R':  3, 'S':  0, 'T':  0, 'V': -2, 'W': -3, 'Y': -4},
    'L': {'A': -2, 'C': -6, 'D': -4, 'E': -3, 'F':  2, 'G': -4, 'H': -2, 'I':  2, 'K': -3, 'L':  6, 'M':  4, 'N': -3, 'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V':  2, 'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -5, 'D': -3, 'E': -2, 'F':  0, 'G': -3, 'H': -2, 'I':  2, 'K':  0, 'L':  4, 'M':  6, 'N': -2, 'P': -2, 'Q': -1, 'R':  0, 'S': -2, 'T': -1, 'V':  2, 'W': -4, 'Y': -2},
    'N': {'A':  0, 'C': -4, 'D':  2, 'E': 1, 'F': -3, 'G':  0, 'H':  2, 'I': -2, 'K':  1, 'L': -3, 'M': -2, 'N':  2, 'P':  0, 'Q':  1, 'R':  0, 'S':  1, 'T':  0, 'V': -2, 'W': -4, 'Y': -2},
    'P': {'A':  1, 'C': -3, 'D': -1, 'E': -1, 'F': -5, 'G':  0, 'H':  0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N':  0, 'P':  6, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -1, 'W': -6, 'Y': -5},
    'Q': {'A':  0, 'C': -5, 'D':  2, 'E': 2, 'F': -5, 'G': -1, 'H':  3, 'I': -2, 'K':  1, 'L': -2, 'M': -1, 'N':  1, 'P':  0, 'Q':  4, 'R':  1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
    'R': {'A': -2, 'C': -4, 'D': -1, 'E': -1, 'F': -4, 'G': -3, 'H':  2, 'I': -2, 'K':  3, 'L': -3, 'M':  0, 'N':  0, 'P':  0, 'Q':  1, 'R':  6, 'S':  0, 'T': -1, 'V': -2, 'W':  2, 'Y': -4},
    'S': {'A':  1, 'C':  0, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P':  1, 'Q': -1, 'R':  0, 'S':  2, 'T':  1, 'V': -1, 'W': -2, 'Y': -3},
    'T': {'A':  1, 'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  0, 'H': -1, 'I':  0, 'K':  0, 'L': -2, 'M': -1, 'N':  0, 'P':  0, 'Q': -1, 'R': -1, 'S':  1, 'T':  3, 'V':  0, 'W': -5, 'Y': -3},
    'V': {'A':  0, 'C': -2, 'D': -2, 'E': -2, 'F': -1, 'G': -1, 'H': -2, 'I':  4, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -6, 'Y': -2},
    'W': {'A': -6, 'C': -8, 'D': -7, 'E': -7, 'F':  0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4, 'P': -6, 'Q': -5, 'R':  2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y':  0},
    'Y': {'A': -3, 'C':  0, 'D': -4, 'E': -4, 'F':  7, 'G': -5, 'H':  0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2, 'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W':  0, 'Y': 10}
}

########################################################################################################################


def Action(Check, Str1, Str2, i, j):
    global rev_seq1, rev_seq2
    if Check == 0:
        Str1 += rev_seq1[-j - 1]
        Str2 += rev_seq2[-i - 1]
        i -= 1
        j -= 1

    elif Check == 1:
        Str1 += "-"
        Str2 += rev_seq2[-i - 1]
        i -= 1

    elif Check == 2:
        Str1 += rev_seq1[-j - 1]
        Str2 += "-"
        j -= 1

    return Str1, Str2, i, j


def TraceBack(Trace_Back_Mat):
    global Score, Out1, Out2, rev_seq1, rev_seq2, len_seq1, len_seq2, Last_Col, Last_Row
    for i in range(len_seq2):
        if Last_Col[len_seq2 - i] == Score:

            loc = [len_seq2 - i - 1, len_seq1 - 1]

            Output1 = "-" * i
            Output2 = rev_seq2[0:i]

            Temp = Trace_Back_Mat

            while(loc[0] != 0 and loc[1] != 0):
                Check = Trace_Back_Mat[loc[0]][loc[1]]

                if Check == 0:
                    Output1, Output2, loc[0], loc[1] = Action(
                        0, Output1, Output2, loc[0], loc[1])

                elif Check == 1:
                    Output1, Output2, loc[0], loc[1] = Action(
                        1, Output1, Output2, loc[0], loc[1])

                elif Check == 2:
                    Output1, Output2, loc[0], loc[1] = Action(
                        2, Output1, Output2, loc[0], loc[1])

                elif Check == 10:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                elif Check == 20:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

                elif Check == 30:
                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

                elif Check == 23:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

            Out1.append(Output1)
            Out2.append(Output2)

    for j in range(len_seq1):
        if Last_Row[len_seq1 - j] == Score:

            loc = [len_seq2 - 1, len_seq1 - j - 1]
            Output1 = rev_seq1[0:j]
            Output2 = "-" * j

            Temp = Trace_Back_Mat

            while(loc[0] != 0 and loc[1] != 0):
                Check = Trace_Back_Mat[loc[0]][loc[1]]

                if Check == 0:
                    Output1, Output2, loc[0], loc[1] = Action(
                        0, Output1, Output2, loc[0], loc[1])

                elif Check == 1:
                    Output1, Output2, loc[0], loc[1] = Action(
                        1, Output1, Output2, loc[0], loc[1])

                elif Check == 2:
                    Output1, Output2, loc[0], loc[1] = Action(
                        2, Output1, Output2, loc[0], loc[1])

                elif Check == 10:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                elif Check == 20:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

                elif Check == 30:
                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

                elif Check == 23:
                    Temp[loc[0]][loc[1]] = 0
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 1
                    TraceBack(Temp)

                    Temp[loc[0]][loc[1]] = 2
                    TraceBack(Temp)

            Out1.append(Output1)
            Out2.append(Output2)

########################################################################################################################


seq1 = input()
seq2 = input()

len_seq1 = len(seq1)
len_seq2 = len(seq2)
gap = -9

Score_Mat = np.zeros((len_seq2 + 1, len_seq1 + 1))
Trace_Back = np.zeros((len_seq2, len_seq1))

for i in range(len_seq2):
    for j in range(len_seq1):
        a = [Score_Mat[i][j] + PAM250[seq2[i]][seq1[j]],
             Score_Mat[i][j+1] + gap, Score_Mat[i+1][j] + gap]
        Score_Mat[i+1][j+1] = max(a)
        Trace_Back[i][j] = (a.index(Score_Mat[i+1][j+1]))

        if_complex = 0
        Temp_check = Trace_Back[i][j]
        for k in range(len(a)):
            if a[k] == Score_Mat[i+1][j+1] and k != Temp_check:
                if_complex += 1
                if if_complex == 1:
                    Trace_Back[i][j] = (Trace_Back[i][j] + k) * 10
                elif if_complex == 2:
                    Trace_Back[i][j] += 23


Last_Row = Score_Mat[len_seq2][:]
Last_Col = Score_Mat[:, len_seq1]
Score = max(max(Last_Row), max(Last_Col))

rev_seq1 = seq1[::-1]
rev_seq2 = seq2[::-1]
Out1 = list()
Out2 = list()

TraceBack(Trace_Back)

for i in range(len(Out1)):
    a = len(Out1[i])
    b = len(Out2[i])
    Out2[i] += rev_seq2[b - Out2[i].count('-'):]
    Out1[i] += rev_seq1[a - Out1[i].count('-'):]
    a = len(Out1[i])
    b = len(Out2[i])
    if a > b:
        Out2[i] += "-" * (a - b)
    elif b > a:
        Out1[i] += "-" * (b - a)

    Out2[i] = Out2[i][::-1]
    Out1[i] = Out1[i][::-1]

print(int(Score))
Test1 = list()
Test2 = list()

Totall_Seq = list()

Test1.append(Out1[0])
Test2.append(Out2[0])

for i in range(len(Out1)):
    if i != 0:
        if (Out1[i] in Test1) and (Out2[i] in Test2):
            continue

    Totall_Seq.append((Out1[i], Out2[i]))
    Test1.append(Out1[i])
    Test2.append(Out2[i])


sortedSeq = [i[0]+i[1] for i in Totall_Seq]
sortedSeq.sort()
for i in sortedSeq:
    print(i[0:int(len(i)/2)])
    print(i[int(len(i)/2):])
