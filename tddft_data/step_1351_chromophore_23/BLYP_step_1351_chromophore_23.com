%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1351_chromophore_23 TDDFT with blyp functional

0 1
Mg   -8.025   40.937   42.569
C   -6.872   37.567   41.758
C   -5.392   41.978   40.730
C   -9.312   43.960   42.710
C   -10.763   39.477   44.077
N   -6.169   39.779   41.561
C   -6.041   38.518   41.228
C   -4.940   38.271   40.227
C   -4.339   39.727   40.049
C   -5.341   40.578   40.847
C   -2.821   39.975   40.387
C   -5.502   37.653   38.873
C   -4.801   36.357   38.324
C   -3.770   36.630   37.244
O   -2.798   37.382   37.267
O   -4.141   35.897   36.098
N   -7.366   42.756   42.002
C   -6.278   42.946   41.235
C   -6.197   44.448   40.941
C   -7.266   45.110   41.602
C   -8.016   43.944   42.136
C   -5.090   45.139   40.204
C   -7.526   46.590   41.659
O   -6.669   47.354   41.274
C   -8.815   47.080   42.341
N   -9.784   41.616   43.237
C   -10.141   42.920   43.227
C   -11.620   43.099   43.632
C   -11.995   41.711   44.226
C   -10.759   40.885   43.870
C   -12.064   44.347   44.491
C   -13.250   40.996   43.742
C   -14.144   40.424   44.879
N   -8.704   38.909   42.953
C   -9.854   38.526   43.648
C   -9.777   37.124   43.651
C   -8.674   36.681   42.873
C   -8.058   37.806   42.495
C   -10.829   36.308   44.268
C   -7.994   35.539   42.365
O   -8.240   34.380   42.584
C   -6.803   36.101   41.601
C   -5.627   35.376   42.153
O   -5.228   35.468   43.360
O   -5.125   34.489   41.242
C   -3.889   33.891   41.447
C   -3.572   36.349   34.834
C   -4.668   36.335   33.736
C   -5.348   37.394   33.274
C   -5.240   38.718   34.024
C   -6.402   37.240   32.142
C   -6.133   38.029   30.822
C   -5.707   37.052   29.707
C   -6.425   37.161   28.326
C   -5.624   36.522   27.241
C   -7.886   36.855   28.440
C   -8.888   38.095   28.228
C   -9.925   38.268   29.372
C   -9.967   39.746   29.956
C   -9.994   39.753   31.495
C   -11.124   40.631   29.313
C   -10.538   41.786   28.489
C   -11.035   43.197   28.984
C   -9.806   44.171   29.123
C   -9.766   44.788   30.476
C   -9.674   45.385   28.163
H   -4.551   42.422   40.194
H   -9.782   44.945   42.761
H   -11.691   39.065   44.481
H   -4.151   37.679   40.690
H   -4.347   40.080   39.018
H   -2.094   40.353   39.668
H   -2.386   39.019   40.679
H   -2.799   40.712   41.190
H   -5.355   38.282   37.996
H   -6.547   37.345   38.901
H   -5.470   35.605   37.907
H   -4.228   35.768   39.040
H   -4.551   44.457   39.546
H   -4.343   45.513   40.903
H   -5.414   45.963   39.568
H   -8.985   46.918   43.405
H   -9.624   46.686   41.726
H   -8.823   48.169   42.291
H   -12.094   43.213   42.657
H   -12.043   41.896   45.299
H   -11.224   45.005   44.715
H   -12.561   43.985   45.391
H   -12.750   44.988   43.938
H   -13.146   40.151   43.062
H   -13.895   41.729   43.259
H   -14.997   41.085   45.031
H   -13.683   40.465   45.866
H   -14.373   39.372   44.710
H   -10.626   35.240   44.188
H   -11.820   36.610   43.929
H   -10.801   36.460   45.347
H   -7.093   35.772   40.603
H   -3.511   33.610   40.464
H   -3.899   32.957   42.009
H   -3.234   34.596   41.959
H   -2.693   35.722   34.684
H   -3.202   37.373   34.781
H   -4.802   35.349   33.293
H   -5.150   38.702   35.110
H   -6.003   39.448   33.755
H   -4.307   39.209   33.747
H   -7.304   37.674   32.572
H   -6.698   36.202   31.990
H   -5.400   38.782   31.111
H   -7.111   38.457   30.603
H   -5.907   36.027   30.017
H   -4.662   37.359   29.747
H   -6.313   38.228   28.131
H   -4.646   36.236   27.629
H   -5.527   37.282   26.466
H   -6.225   35.731   26.792
H   -8.115   36.280   29.337
H   -8.182   36.203   27.617
H   -9.528   37.917   27.363
H   -8.352   38.994   27.926
H   -9.789   37.648   30.258
H   -10.899   37.993   28.970
H   -9.017   40.195   29.664
H   -10.397   38.867   31.987
H   -10.638   40.569   31.823
H   -8.946   39.876   31.768
H   -11.757   41.094   30.071
H   -11.704   39.991   28.649
H   -10.877   41.761   27.453
H   -9.460   41.710   28.349
H   -11.533   43.189   29.954
H   -11.750   43.558   28.244
H   -8.876   43.619   28.983
H   -8.712   44.895   30.731
H   -10.260   44.270   31.298
H   -10.308   45.733   30.442
H   -10.670   45.716   27.866
H   -9.282   45.060   27.199
H   -8.982   46.171   28.464

