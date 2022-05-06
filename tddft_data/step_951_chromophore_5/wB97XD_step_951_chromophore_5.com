%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_951_chromophore_5 TDDFT with wB97XD functional

0 1
Mg   24.900   -7.596   46.360
C   26.959   -4.963   45.397
C   22.626   -6.715   44.042
C   23.042   -10.388   47.213
C   27.614   -9.051   48.080
N   24.881   -6.112   44.762
C   25.774   -5.070   44.596
C   25.314   -4.101   43.524
C   23.787   -4.499   43.395
C   23.770   -5.920   43.989
C   22.735   -3.603   44.034
C   26.119   -4.202   42.219
C   26.646   -2.873   41.683
C   26.525   -2.587   40.224
O   27.460   -2.409   39.448
O   25.208   -2.504   39.895
N   22.935   -8.295   45.824
C   22.237   -7.846   44.813
C   20.960   -8.604   44.667
C   21.037   -9.599   45.708
C   22.405   -9.498   46.288
C   19.826   -8.294   43.734
C   19.899   -10.537   46.152
O   18.872   -10.422   45.493
C   19.955   -11.560   47.262
N   25.303   -9.564   47.464
C   24.312   -10.463   47.750
C   24.843   -11.602   48.654
C   26.385   -11.180   48.781
C   26.465   -9.794   48.139
C   24.067   -11.725   50.001
C   27.248   -12.248   48.140
C   28.555   -12.353   48.851
N   26.934   -7.123   46.671
C   27.879   -7.760   47.520
C   29.115   -6.896   47.623
C   28.735   -5.766   46.870
C   27.442   -5.963   46.287
C   30.455   -7.144   48.329
C   29.243   -4.474   46.370
O   30.326   -3.984   46.527
C   28.034   -3.974   45.407
C   27.558   -2.659   45.881
O   27.950   -1.575   45.497
O   26.566   -2.816   46.859
C   25.808   -1.694   47.462
C   24.914   -2.532   38.416
C   24.680   -3.950   37.951
C   25.181   -4.587   36.875
C   26.071   -4.041   35.815
C   24.694   -6.028   36.586
C   23.235   -6.263   36.159
C   23.070   -7.722   35.633
C   22.786   -7.691   34.109
C   21.841   -8.873   33.708
C   24.141   -7.782   33.275
C   24.176   -6.964   31.924
C   25.357   -5.973   31.944
C   24.988   -4.781   31.005
C   24.316   -3.704   31.867
C   26.336   -4.166   30.401
C   26.259   -3.658   28.876
C   26.270   -4.807   27.911
C   25.386   -4.443   26.673
C   23.857   -4.705   27.101
C   25.797   -5.334   25.399
H   21.946   -6.293   43.299
H   22.442   -11.247   47.519
H   28.454   -9.451   48.651
H   25.372   -3.099   43.949
H   23.533   -4.569   42.338
H   22.141   -3.217   43.206
H   23.187   -2.799   44.616
H   22.069   -4.269   44.583
H   25.500   -4.728   41.493
H   26.876   -4.959   42.422
H   27.704   -2.710   41.890
H   26.133   -2.041   42.164
H   18.847   -8.520   44.157
H   20.109   -8.892   42.869
H   19.823   -7.242   43.447
H   20.458   -11.287   48.190
H   20.553   -12.332   46.777
H   19.013   -12.084   47.425
H   24.732   -12.519   48.074
H   26.600   -11.123   49.848
H   23.071   -11.386   49.719
H   24.549   -11.027   50.685
H   24.229   -12.750   50.337
H   27.452   -11.972   47.105
H   26.744   -13.214   48.120
H   29.367   -11.833   48.342
H   28.913   -13.378   48.948
H   28.621   -11.928   49.852
H   31.262   -6.633   47.804
H   30.650   -8.216   48.357
H   30.480   -6.968   49.405
H   28.394   -3.803   44.392
H   24.810   -2.072   47.684
H   25.823   -0.721   46.972
H   26.155   -1.519   48.480
H   25.728   -2.079   37.851
H   24.077   -1.875   38.180
H   24.056   -4.548   38.614
H   27.025   -4.558   35.914
H   26.177   -2.959   35.899
H   25.570   -4.225   34.864
H   24.963   -6.682   37.416
H   25.341   -6.383   35.784
H   22.952   -5.573   35.364
H   22.569   -5.888   36.936
H   22.201   -8.037   36.211
H   24.032   -8.159   35.901
H   22.298   -6.753   33.846
H   22.140   -9.527   32.889
H   20.845   -8.565   33.393
H   21.617   -9.597   34.492
H   24.389   -8.789   32.937
H   25.002   -7.552   33.903
H   23.280   -6.354   32.038
H   24.264   -7.603   31.045
H   26.169   -6.551   31.504
H   25.655   -5.660   32.944
H   24.317   -5.151   30.230
H   24.648   -2.671   31.756
H   24.199   -4.076   32.884
H   23.263   -3.563   31.619
H   27.083   -4.960   30.418
H   26.647   -3.379   31.087
H   27.181   -3.093   28.737
H   25.408   -2.997   28.708
H   25.964   -5.740   28.385
H   27.318   -4.796   27.613
H   25.471   -3.397   26.378
H   23.613   -5.709   26.752
H   23.138   -4.078   26.574
H   23.697   -4.644   28.177
H   26.464   -6.150   25.678
H   26.398   -4.732   24.717
H   25.021   -5.825   24.812
