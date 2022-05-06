%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1501_chromophore_27 TDDFT with wB97XD functional

0 1
Mg   -4.398   24.642   26.832
C   -3.060   26.632   29.165
C   -5.773   22.612   29.279
C   -5.774   22.601   24.352
C   -2.733   26.360   24.239
N   -4.208   24.530   28.966
C   -3.773   25.595   29.723
C   -4.155   25.346   31.226
C   -4.528   23.852   31.216
C   -4.909   23.621   29.695
C   -3.366   22.936   31.576
C   -5.374   26.246   31.635
C   -5.349   26.770   33.125
C   -6.394   26.283   33.999
O   -7.396   26.899   34.346
O   -6.073   25.080   34.446
N   -5.710   22.915   26.836
C   -6.158   22.289   27.933
C   -6.955   21.158   27.594
C   -7.032   21.121   26.174
C   -6.139   22.250   25.683
C   -7.539   20.137   28.546
C   -7.808   20.145   25.294
O   -7.651   20.141   24.108
C   -8.750   19.117   25.927
N   -4.279   24.508   24.567
C   -5.040   23.658   23.826
C   -4.761   23.814   22.314
C   -3.754   25.014   22.295
C   -3.562   25.341   23.794
C   -4.180   22.496   21.572
C   -4.073   26.217   21.411
C   -5.350   26.952   21.841
N   -3.139   26.243   26.664
C   -2.452   26.804   25.547
C   -1.703   27.897   26.042
C   -1.947   27.953   27.451
C   -2.782   26.866   27.748
C   -0.954   28.825   25.119
C   -1.805   28.637   28.718
O   -1.316   29.714   28.980
C   -2.500   27.818   29.862
C   -1.541   27.390   30.867
O   -0.831   26.394   30.798
O   -1.580   28.292   31.938
C   -0.840   27.796   33.152
C   -6.616   24.644   35.757
C   -7.782   23.728   35.668
C   -8.129   22.679   36.425
C   -7.291   22.281   37.630
C   -9.260   21.737   36.045
C   -10.599   22.187   36.653
C   -11.195   21.211   37.729
C   -12.354   20.281   37.288
C   -13.690   20.979   37.602
C   -12.267   18.902   38.018
C   -11.190   17.956   37.482
C   -10.188   17.559   38.633
C   -8.737   17.355   38.143
C   -7.903   18.601   38.423
C   -8.147   16.105   38.801
C   -7.015   15.474   37.823
C   -5.490   15.698   38.263
C   -4.764   14.287   38.584
C   -4.365   14.241   40.052
C   -3.520   14.253   37.648
H   -6.122   21.967   30.088
H   -6.202   21.921   23.614
H   -2.207   26.973   23.503
H   -3.256   25.595   31.790
H   -5.388   23.811   31.883
H   -3.463   21.975   31.070
H   -3.254   22.778   32.648
H   -2.495   23.497   31.237
H   -6.303   25.770   31.321
H   -5.318   27.191   31.095
H   -5.268   27.856   33.168
H   -4.393   26.541   33.595
H   -8.614   20.274   28.429
H   -7.191   20.259   29.572
H   -7.287   19.101   28.321
H   -9.229   18.577   25.110
H   -9.473   19.662   26.532
H   -8.133   18.428   26.504
H   -5.724   24.091   21.884
H   -2.737   24.703   22.057
H   -4.041   21.749   22.353
H   -3.246   22.752   21.071
H   -4.897   22.132   20.836
H   -4.317   25.700   20.483
H   -3.199   26.822   21.169
H   -5.108   27.781   22.506
H   -5.981   26.368   22.510
H   -5.888   27.245   20.939
H   -0.114   28.369   24.595
H   -0.696   29.685   25.737
H   -1.525   29.094   24.231
H   -3.225   28.454   30.371
H   -0.994   26.726   33.289
H   -1.195   28.400   33.987
H   0.243   27.881   33.063
H   -6.643   25.438   36.504
H   -5.809   24.012   36.128
H   -8.317   23.814   34.721
H   -7.915   22.560   38.479
H   -6.361   22.805   37.850
H   -7.145   21.201   37.655
H   -8.979   20.731   36.358
H   -9.415   21.716   34.966
H   -11.305   22.324   35.834
H   -10.309   23.151   37.071
H   -11.472   21.794   38.608
H   -10.423   20.566   38.148
H   -12.286   20.211   36.202
H   -14.332   20.756   36.750
H   -13.456   22.043   37.590
H   -14.137   20.688   38.553
H   -13.150   18.294   37.821
H   -12.271   19.036   39.100
H   -10.721   18.303   36.562
H   -11.714   17.104   37.048
H   -10.616   16.593   38.898
H   -10.163   18.209   39.508
H   -8.817   17.252   37.061
H   -8.553   19.418   38.735
H   -7.386   18.986   37.545
H   -7.144   18.496   39.199
H   -8.980   15.406   38.880
H   -7.772   16.296   39.807
H   -7.056   15.931   36.834
H   -7.210   14.410   37.684
H   -5.341   16.405   39.079
H   -5.020   16.249   37.448
H   -5.482   13.504   38.338
H   -4.897   14.803   40.819
H   -3.312   14.471   40.210
H   -4.472   13.211   40.394
H   -2.609   13.844   38.085
H   -3.087   15.218   37.384
H   -3.639   13.623   36.767

