%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_901_chromophore_8 TDDFT with PBE1PBE functional

0 1
Mg   44.776   3.045   47.178
C   42.364   5.531   46.752
C   42.109   0.786   47.099
C   46.973   0.597   47.010
C   47.283   5.453   46.689
N   42.540   3.165   47.095
C   41.812   4.269   46.947
C   40.360   3.897   46.697
C   40.283   2.504   47.416
C   41.720   2.115   47.195
C   39.799   2.427   48.947
C   40.120   3.960   45.183
C   38.688   4.347   44.786
C   38.563   5.470   43.810
O   37.751   6.365   43.885
O   39.461   5.212   42.743
N   44.556   0.962   47.077
C   43.427   0.214   47.019
C   43.733   -1.188   46.966
C   45.177   -1.277   46.952
C   45.664   0.130   46.992
C   42.787   -2.265   46.804
C   45.960   -2.578   46.922
O   45.299   -3.574   47.040
C   47.432   -2.779   46.636
N   46.858   3.000   46.703
C   47.599   1.823   46.952
C   49.125   2.216   47.149
C   49.103   3.759   47.145
C   47.636   4.144   46.841
C   49.849   1.638   48.432
C   50.134   4.301   46.143
C   51.270   5.093   46.893
N   44.866   5.062   46.873
C   45.941   5.931   46.614
C   45.392   7.292   46.391
C   44.021   7.089   46.523
C   43.749   5.754   46.723
C   46.154   8.580   46.139
C   42.802   7.863   46.504
O   42.614   9.085   46.419
C   41.675   6.852   46.689
C   40.965   7.233   47.941
O   41.388   7.314   49.083
O   39.658   7.436   47.595
C   38.634   7.701   48.613
C   39.398   6.162   41.683
C   40.342   5.776   40.519
C   40.570   6.527   39.417
C   40.098   7.989   39.242
C   41.432   6.027   38.352
C   40.841   4.748   37.646
C   41.595   3.465   38.093
C   42.024   2.469   36.909
C   41.876   0.947   37.410
C   43.357   2.718   36.324
C   43.281   3.403   34.944
C   44.463   4.294   34.662
C   45.095   3.929   33.300
C   46.615   3.555   33.565
C   44.865   5.018   32.206
C   43.547   4.625   31.523
C   43.459   5.457   30.179
C   43.428   4.415   28.932
C   41.906   4.025   28.722
C   44.078   5.019   27.690
H   41.253   0.115   47.194
H   47.702   -0.162   47.301
H   47.989   6.263   46.884
H   39.822   4.726   47.156
H   39.695   1.825   46.798
H   40.600   2.059   49.588
H   38.961   1.731   48.976
H   39.483   3.439   49.199
H   40.464   3.001   44.797
H   40.741   4.755   44.770
H   38.018   4.417   45.643
H   38.359   3.487   44.202
H   42.852   -2.671   45.794
H   41.780   -1.920   47.041
H   43.005   -3.121   47.443
H   48.117   -2.422   47.405
H   47.772   -2.252   45.745
H   47.554   -3.816   46.323
H   49.560   1.794   46.243
H   49.231   4.167   48.148
H   50.680   1.064   48.021
H   49.183   0.983   48.995
H   50.274   2.377   49.111
H   49.770   4.921   45.324
H   50.656   3.512   45.602
H   52.035   5.199   46.123
H   51.756   4.637   47.756
H   50.919   6.109   47.070
H   45.517   9.370   45.741
H   47.032   8.379   45.526
H   46.436   9.048   47.082
H   41.002   6.786   45.834
H   38.674   7.012   49.456
H   37.629   7.600   48.202
H   38.776   8.683   49.065
H   39.849   7.116   41.957
H   38.375   6.277   41.323
H   40.779   4.778   40.551
H   39.557   8.439   40.075
H   39.305   7.943   38.495
H   40.983   8.476   38.833
H   42.367   5.782   38.857
H   41.712   6.680   37.526
H   40.997   4.828   36.570
H   39.761   4.638   37.744
H   40.951   3.041   38.863
H   42.487   3.725   38.663
H   41.296   2.597   36.108
H   41.668   0.797   38.469
H   42.769   0.391   37.122
H   41.056   0.559   36.807
H   44.069   1.893   36.300
H   43.966   3.367   36.953
H   42.404   4.049   34.889
H   42.979   2.639   34.228
H   45.234   4.337   35.431
H   44.222   5.354   34.575
H   44.721   2.969   32.944
H   47.170   4.488   33.656
H   47.037   3.073   32.682
H   46.788   3.032   34.505
H   45.736   4.960   31.554
H   44.747   5.978   32.709
H   42.788   4.977   32.222
H   43.355   3.604   31.193
H   44.210   6.244   30.116
H   42.489   5.921   30.362
H   44.157   3.619   29.083
H   41.761   3.016   29.106
H   41.585   4.067   27.681
H   41.237   4.675   29.285
H   44.991   4.429   27.609
H   44.228   6.080   27.894
H   43.459   4.978   26.794

