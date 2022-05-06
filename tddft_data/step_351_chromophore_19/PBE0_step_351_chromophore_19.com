%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_351_chromophore_19 TDDFT with PBE1PBE functional

0 1
Mg   25.558   51.122   26.228
C   23.264   52.294   28.688
C   27.528   49.857   28.745
C   27.918   50.629   23.855
C   23.515   52.908   23.810
N   25.501   51.254   28.469
C   24.424   51.710   29.211
C   24.641   51.259   30.710
C   25.991   50.576   30.725
C   26.427   50.542   29.214
C   27.095   51.219   31.564
C   23.462   50.324   31.263
C   22.824   50.885   32.530
C   23.794   51.500   33.580
O   24.080   52.683   33.628
O   24.359   50.504   34.370
N   27.480   50.301   26.300
C   28.026   49.776   27.391
C   29.287   49.162   27.018
C   29.490   49.361   25.642
C   28.277   50.133   25.171
C   30.212   48.420   27.923
C   30.709   48.930   24.736
O   30.761   49.331   23.602
C   31.865   48.074   25.285
N   25.687   51.856   24.139
C   26.791   51.342   23.435
C   26.588   51.817   21.972
C   25.120   52.306   21.852
C   24.746   52.444   23.368
C   27.674   52.913   21.576
C   24.164   51.236   21.097
C   23.062   51.770   20.173
N   23.759   52.263   26.165
C   23.052   52.870   25.168
C   21.789   53.320   25.711
C   21.859   53.156   27.076
C   23.103   52.564   27.328
C   20.695   54.074   25.001
C   21.207   53.538   28.291
O   20.193   54.207   28.561
C   22.041   52.988   29.418
C   22.513   54.152   30.332
O   23.487   54.836   30.056
O   21.669   54.315   31.486
C   22.137   55.300   32.469
C   25.117   50.882   35.521
C   24.840   50.064   36.829
C   25.330   50.294   38.032
C   26.279   51.480   38.399
C   24.919   49.518   39.296
C   25.580   48.096   39.357
C   26.474   47.961   40.549
C   26.501   46.510   41.170
C   26.783   45.453   40.086
C   25.230   46.349   42.057
C   25.719   45.933   43.515
C   25.455   46.988   44.637
C   26.720   47.310   45.586
C   27.182   46.093   46.332
C   26.349   48.544   46.474
C   27.429   49.598   46.433
C   26.959   50.944   45.780
C   27.662   51.145   44.404
C   29.110   51.760   44.525
C   26.790   51.833   43.394
H   28.050   49.327   29.546
H   28.533   50.330   23.004
H   22.777   53.348   23.135
H   24.657   52.197   31.265
H   25.857   49.545   31.052
H   27.580   50.431   32.141
H   26.650   51.979   32.205
H   27.948   51.707   31.092
H   23.835   49.321   31.472
H   22.744   50.133   30.465
H   22.369   50.000   32.976
H   22.072   51.589   32.173
H   31.229   48.797   28.030
H   30.306   47.388   27.583
H   29.645   48.365   28.852
H   31.488   47.106   25.614
H   32.330   48.556   26.145
H   32.521   47.959   24.422
H   26.766   50.952   21.333
H   25.112   53.286   21.374
H   27.760   53.627   22.396
H   27.340   53.532   20.743
H   28.602   52.432   21.267
H   23.681   50.644   21.875
H   24.790   50.625   20.447
H   22.166   51.395   20.668
H   23.176   51.345   19.175
H   23.186   52.851   20.111
H   20.017   54.579   25.689
H   20.159   53.387   24.346
H   21.090   54.903   24.413
H   21.454   52.294   30.020
H   22.842   56.108   32.270
H   22.450   54.765   33.366
H   21.369   55.976   32.846
H   24.842   51.873   35.880
H   26.176   50.821   35.270
H   24.061   49.308   36.737
H   25.782   52.157   39.093
H   26.578   52.020   37.500
H   27.221   51.141   38.830
H   23.873   49.233   39.404
H   25.214   50.075   40.185
H   26.299   47.975   38.547
H   24.779   47.357   39.308
H   26.185   48.688   41.308
H   27.501   48.216   40.289
H   27.406   46.522   41.778
H   26.362   45.729   39.119
H   26.493   44.464   40.441
H   27.870   45.387   40.048
H   24.758   45.506   41.552
H   24.631   47.259   42.083
H   26.776   45.666   43.518
H   25.133   45.031   43.695
H   24.597   46.603   45.187
H   25.105   47.856   44.078
H   27.438   47.659   44.844
H   26.355   45.930   47.023
H   28.144   46.361   46.768
H   27.301   45.244   45.658
H   26.138   48.327   47.521
H   25.414   49.027   46.192
H   28.275   49.292   45.818
H   27.715   49.791   47.466
H   27.234   51.794   46.406
H   25.890   50.930   45.570
H   27.931   50.203   43.926
H   29.572   51.779   43.537
H   29.722   51.093   45.132
H   29.080   52.716   45.048
H   26.832   52.904   43.593
H   25.753   51.501   43.430
H   27.150   51.691   42.375

