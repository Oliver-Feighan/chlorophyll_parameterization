%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1_chromophore_3 TDDFT with PBE1PBE functional

0 1
Mg   1.397   7.672   26.334
C   1.724   9.741   29.051
C   2.013   4.991   28.381
C   1.182   5.575   23.580
C   1.222   10.435   24.247
N   1.881   7.369   28.504
C   1.860   8.397   29.403
C   2.088   7.914   30.753
C   2.400   6.433   30.607
C   2.122   6.205   29.057
C   3.813   5.913   31.044
C   0.869   8.215   31.690
C   1.184   8.506   33.198
C   2.448   7.854   33.792
O   3.525   8.347   34.183
O   2.178   6.495   34.023
N   1.466   5.549   26.075
C   1.719   4.629   27.078
C   1.707   3.272   26.495
C   1.454   3.427   25.064
C   1.331   4.879   24.847
C   1.961   1.993   27.276
C   1.226   2.335   24.045
O   0.942   2.650   22.876
C   1.349   0.883   24.320
N   1.304   7.948   24.201
C   1.170   6.942   23.332
C   0.899   7.477   21.916
C   0.648   9.069   22.124
C   1.006   9.146   23.650
C   2.063   7.120   20.941
C   -0.806   9.555   21.874
C   -1.005   10.456   20.646
N   1.524   9.708   26.541
C   1.451   10.734   25.596
C   1.565   12.050   26.263
C   1.690   11.659   27.617
C   1.669   10.243   27.729
C   1.493   13.371   25.672
C   1.738   12.190   28.995
O   1.682   13.313   29.421
C   1.805   10.957   29.965
C   2.974   11.120   30.844
O   4.122   10.981   30.392
O   2.681   11.286   32.163
C   3.771   11.252   33.133
C   3.048   5.855   35.050
C   1.929   5.252   35.920
C   1.869   5.066   37.262
C   2.844   5.658   38.257
C   0.649   4.441   37.846
C   0.829   3.116   38.637
C   0.458   1.834   37.761
C   0.426   0.430   38.301
C   -0.691   -0.398   37.669
C   1.726   -0.354   38.157
C   2.354   -0.963   39.380
C   3.109   -2.224   38.988
C   3.987   -2.755   40.173
C   3.117   -3.186   41.378
C   5.137   -1.851   40.639
C   6.399   -2.671   41.039
C   7.463   -1.796   41.848
C   8.896   -2.176   41.372
C   9.882   -2.039   42.589
C   9.265   -1.108   40.271
H   2.211   4.159   29.060
H   1.154   4.901   22.721
H   1.034   11.270   23.569
H   2.966   8.427   31.147
H   1.620   5.817   31.056
H   4.563   6.577   31.475
H   4.274   5.435   30.180
H   3.771   5.074   31.739
H   0.125   7.425   31.587
H   0.273   9.011   31.244
H   0.337   8.213   33.819
H   1.413   9.554   33.389
H   2.028   2.150   28.352
H   2.913   1.543   26.993
H   1.087   1.344   27.224
H   1.036   0.372   23.410
H   0.748   0.547   25.166
H   2.368   0.659   24.636
H   -0.069   7.080   21.610
H   1.359   9.680   21.568
H   1.725   6.673   20.006
H   2.747   6.405   21.399
H   2.592   8.064   20.808
H   -1.226   10.107   22.714
H   -1.498   8.721   21.755
H   -0.249   10.220   19.897
H   -0.863   11.489   20.962
H   -2.045   10.342   20.343
H   1.264   14.148   26.401
H   0.762   13.444   24.867
H   2.505   13.659   25.387
H   0.852   11.055   30.486
H   4.474   12.081   33.047
H   4.270   10.283   33.126
H   3.365   11.366   34.138
H   3.669   6.537   35.629
H   3.645   5.042   34.637
H   1.141   4.853   35.281
H   2.241   6.388   38.797
H   3.667   6.079   37.680
H   3.186   4.837   38.888
H   -0.121   4.345   37.080
H   0.245   5.038   38.663
H   0.325   3.045   39.601
H   1.867   2.982   38.940
H   1.070   1.662   36.876
H   -0.538   2.056   37.377
H   0.219   0.481   39.370
H   -1.049   0.006   36.722
H   -1.532   -0.179   38.326
H   -0.450   -1.431   37.415
H   2.424   0.420   37.837
H   1.644   -1.051   37.322
H   1.593   -1.066   40.153
H   3.027   -0.185   39.739
H   3.572   -2.108   38.008
H   2.339   -2.994   38.930
H   4.472   -3.626   39.732
H   2.350   -3.916   41.121
H   2.586   -2.314   41.762
H   3.682   -3.709   42.149
H   4.662   -1.306   41.455
H   5.377   -1.127   39.860
H   6.758   -2.976   40.056
H   5.986   -3.522   41.582
H   7.254   -2.036   42.890
H   7.172   -0.748   41.771
H   9.044   -3.111   40.833
H   9.386   -1.986   43.558
H   10.420   -1.092   42.540
H   10.532   -2.913   42.553
H   9.459   -1.773   39.430
H   10.177   -0.553   40.493
H   8.540   -0.302   40.162

