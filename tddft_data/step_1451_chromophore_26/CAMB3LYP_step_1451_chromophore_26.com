%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1451_chromophore_26 TDDFT with cam-b3lyp functional

0 1
Mg   -9.181   17.752   42.468
C   -5.729   17.258   42.660
C   -8.546   21.123   41.736
C   -12.503   18.238   41.794
C   -9.653   14.333   42.511
N   -7.361   19.054   42.141
C   -6.037   18.559   42.355
C   -5.041   19.707   42.043
C   -5.976   20.931   41.983
C   -7.412   20.351   42.006
C   -5.794   22.006   43.142
C   -4.185   19.717   40.706
C   -4.940   19.109   39.514
C   -4.369   19.623   38.160
O   -3.226   19.486   37.741
O   -5.307   20.282   37.372
N   -10.380   19.463   41.871
C   -9.920   20.758   41.654
C   -11.015   21.585   41.328
C   -12.207   20.791   41.398
C   -11.753   19.423   41.799
C   -10.824   23.056   40.851
C   -13.569   21.211   41.169
O   -13.759   22.341   40.724
C   -14.807   20.428   41.570
N   -10.872   16.410   42.331
C   -12.140   16.855   42.035
C   -13.096   15.717   41.616
C   -12.097   14.502   41.687
C   -10.782   15.080   42.165
C   -14.281   15.641   42.589
C   -11.986   13.667   40.400
C   -12.754   12.345   40.377
N   -7.974   16.056   42.671
C   -8.293   14.701   42.769
C   -7.110   13.934   42.896
C   -6.063   14.888   42.903
C   -6.653   16.172   42.752
C   -7.027   12.411   42.941
C   -4.622   15.100   42.889
O   -3.741   14.253   42.817
C   -4.413   16.642   42.846
C   -3.657   16.978   44.115
O   -2.508   17.284   44.206
O   -4.505   17.008   45.179
C   -4.006   17.502   46.411
C   -4.994   20.815   36.039
C   -6.028   20.476   35.035
C   -6.207   19.376   34.353
C   -5.286   18.152   34.533
C   -7.120   19.322   33.160
C   -8.499   18.819   33.600
C   -9.239   17.981   32.451
C   -10.758   18.151   32.465
C   -11.257   16.830   31.920
C   -11.341   19.238   31.441
C   -11.244   20.728   31.942
C   -10.047   21.621   31.495
C   -10.437   22.930   30.688
C   -9.923   24.281   31.351
C   -10.068   22.897   29.137
C   -11.335   22.621   28.204
C   -11.522   23.865   27.186
C   -11.786   23.280   25.802
C   -10.437   22.972   25.063
C   -12.798   24.081   24.999
H   -8.317   22.173   41.545
H   -13.579   18.272   41.610
H   -10.014   13.314   42.661
H   -4.415   19.761   42.933
H   -5.888   21.461   41.034
H   -6.729   22.110   43.693
H   -5.527   23.000   42.785
H   -5.048   21.613   43.833
H   -3.231   19.198   40.797
H   -3.977   20.741   40.394
H   -6.017   19.233   39.621
H   -4.697   18.047   39.487
H   -11.175   23.301   39.849
H   -9.768   23.294   40.725
H   -11.234   23.638   41.677
H   -14.871   19.892   42.517
H   -15.031   19.715   40.776
H   -15.585   21.190   41.522
H   -13.493   15.890   40.616
H   -12.460   13.796   42.434
H   -14.208   16.273   43.474
H   -14.256   14.589   42.873
H   -15.194   15.875   42.041
H   -10.926   13.454   40.262
H   -12.379   14.268   39.579
H   -13.147   12.114   39.387
H   -13.619   12.313   41.039
H   -12.216   11.435   40.644
H   -7.952   11.968   42.574
H   -6.728   12.156   43.958
H   -6.165   12.281   42.286
H   -3.804   16.939   41.992
H   -4.758   17.530   47.199
H   -3.779   18.564   46.321
H   -3.127   16.928   46.706
H   -4.037   20.436   35.680
H   -4.959   21.880   36.266
H   -6.786   21.224   34.801
H   -6.017   17.346   34.610
H   -4.713   18.017   35.450
H   -4.671   17.880   33.675
H   -6.783   18.529   32.493
H   -7.273   20.238   32.589
H   -9.099   19.707   33.801
H   -8.421   18.200   34.494
H   -8.951   16.939   32.319
H   -8.871   18.436   31.531
H   -11.195   18.134   33.463
H   -10.739   16.448   31.041
H   -12.288   16.891   31.570
H   -11.119   16.082   32.701
H   -12.391   19.017   31.252
H   -10.867   19.129   30.465
H   -11.209   20.738   33.032
H   -12.152   21.269   31.676
H   -9.353   20.991   30.937
H   -9.538   21.856   32.429
H   -11.500   23.071   30.883
H   -8.998   24.682   30.937
H   -9.831   24.065   32.415
H   -10.716   25.013   31.198
H   -9.247   22.193   28.998
H   -9.661   23.872   28.870
H   -12.198   22.452   28.847
H   -11.070   21.699   27.687
H   -10.661   24.530   27.253
H   -12.465   24.340   27.456
H   -12.339   22.348   25.919
H   -10.390   21.889   24.953
H   -9.642   23.345   25.709
H   -10.324   23.491   24.111
H   -13.369   24.811   25.573
H   -13.449   23.459   24.384
H   -12.190   24.703   24.342

