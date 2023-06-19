import TM_Exact_Solution as TM
import Method_of_Integral_Relations as MIR

# Generate Data for Table 5.1
Mach = [1.5, 2, 4, 8, 10]

for M in Mach:

    # OneStrip Solution
    MIR_1 = MIR.Frozen_Cone_Flow(M, 1.4, 45)
    Cone_Angle_1st = MIR.Frozen_Cone_Flow.one_strip_solve(MIR_1, full_output=True)

    # Exact Solution
    TM.Taylor_Maccoll_Solve_All(M, 1.4, 45)