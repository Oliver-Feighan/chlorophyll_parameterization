%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_301_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.394   24.699   26.546
C   -3.712   26.462   29.261
C   -6.220   22.368   28.925
C   -6.809   22.859   24.192
C   -3.984   26.907   24.354
N   -5.159   24.557   28.820
C   -4.497   25.358   29.747
C   -4.665   24.950   31.195
C   -5.339   23.571   30.986
C   -5.681   23.517   29.452
C   -4.495   22.282   31.498
C   -5.381   26.067   32.039
C   -6.709   25.849   32.817
C   -6.648   25.287   34.287
O   -5.671   25.372   35.057
O   -7.941   24.880   34.597
N   -6.407   22.889   26.503
C   -6.546   22.068   27.579
C   -6.994   20.806   27.183
C   -7.338   20.972   25.810
C   -6.923   22.338   25.421
C   -7.096   19.684   28.162
C   -8.035   20.010   24.878
O   -8.194   20.282   23.710
C   -8.609   18.722   25.388
N   -5.368   24.773   24.591
C   -6.110   24.021   23.730
C   -6.143   24.564   22.268
C   -5.236   25.854   22.363
C   -4.874   25.921   23.943
C   -5.584   23.569   21.204
C   -5.912   27.147   21.957
C   -5.033   27.864   20.892
N   -4.125   26.385   26.708
C   -3.549   27.199   25.702
C   -2.758   28.250   26.338
C   -2.810   27.969   27.692
C   -3.612   26.784   27.911
C   -2.114   29.348   25.524
C   -2.270   28.384   28.866
O   -1.334   29.183   29.065
C   -2.842   27.457   29.999
C   -1.732   26.837   30.807
O   -0.899   26.036   30.492
O   -1.734   27.414   32.062
C   -0.930   26.789   33.159
C   -8.067   24.273   35.876
C   -8.986   23.067   35.768
C   -9.235   22.210   36.751
C   -8.787   22.393   38.209
C   -10.241   21.159   36.514
C   -11.730   21.471   36.882
C   -12.353   20.495   37.928
C   -12.915   19.147   37.357
C   -14.416   19.431   36.880
C   -12.721   17.928   38.249
C   -11.513   17.108   37.811
C   -10.701   16.685   39.055
C   -9.157   17.109   38.906
C   -8.856   17.959   40.115
C   -8.279   15.844   38.856
C   -7.099   16.241   37.920
C   -6.039   15.108   38.127
C   -5.006   15.486   39.199
C   -3.956   16.392   38.521
C   -4.386   14.294   40.007
H   -6.504   21.618   29.666
H   -7.159   22.289   23.329
H   -3.612   27.660   23.657
H   -3.646   24.763   31.534
H   -6.276   23.505   31.539
H   -3.530   22.578   31.908
H   -4.281   21.671   30.621
H   -4.987   21.616   32.206
H   -5.547   26.916   31.376
H   -4.570   26.378   32.696
H   -7.314   25.240   32.145
H   -7.083   26.873   32.818
H   -6.150   19.707   28.702
H   -7.405   18.708   27.788
H   -7.972   19.861   28.787
H   -9.001   18.289   24.468
H   -9.388   19.069   26.068
H   -7.840   18.099   25.845
H   -7.194   24.789   22.086
H   -4.318   25.661   21.809
H   -4.687   23.043   21.534
H   -5.202   23.934   20.251
H   -6.434   22.917   21.005
H   -5.940   27.839   22.798
H   -6.950   26.909   21.725
H   -3.965   27.791   21.097
H   -5.210   28.904   21.165
H   -5.334   27.752   19.851
H   -2.850   29.937   24.977
H   -1.583   28.859   24.707
H   -1.448   29.922   26.168
H   -3.422   28.078   30.681
H   -0.717   27.525   33.934
H   0.046   26.479   32.786
H   -1.467   25.906   33.506
H   -8.539   25.013   36.521
H   -7.189   23.891   36.397
H   -9.509   22.847   34.837
H   -9.680   22.454   38.831
H   -8.238   23.320   38.375
H   -8.164   21.538   38.474
H   -9.989   20.263   37.081
H   -10.275   20.756   35.502
H   -12.235   21.390   35.919
H   -11.982   22.459   37.266
H   -13.202   20.963   38.426
H   -11.524   20.320   38.614
H   -12.451   19.072   36.374
H   -14.849   18.505   36.501
H   -14.386   20.086   36.009
H   -14.913   19.958   37.695
H   -13.569   17.245   38.208
H   -12.623   18.478   39.185
H   -10.885   17.503   37.013
H   -11.777   16.121   37.430
H   -10.812   15.633   39.317
H   -11.144   17.099   39.961
H   -9.072   17.702   37.995
H   -7.776   17.887   40.237
H   -9.484   17.632   40.944
H   -9.276   18.951   39.942
H   -8.927   15.034   38.522
H   -7.882   15.465   39.798
H   -6.806   17.285   38.030
H   -7.473   16.004   36.923
H   -5.559   14.892   37.173
H   -6.595   14.216   38.414
H   -5.446   16.094   39.989
H   -3.794   17.189   39.247
H   -4.295   16.871   37.602
H   -3.064   15.771   38.436
H   -4.435   14.525   41.071
H   -3.324   14.082   39.887
H   -4.963   13.385   39.834
